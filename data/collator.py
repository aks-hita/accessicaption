"""
collator.py
-----------
PyTorch Dataset and DataCollator for the AccessiCaption vision-language
captioning pipeline.

Classes
-------
AccessiCaptionDataset
    Loads processed JSON records, opens images, and tokenizes captions.

DataCollatorForVLMCaptioning
    Collates a batch of ``AccessiCaptionDataset`` items into padded tensors
    suitable for a vision-language model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin  # type: ignore[import]

from wcag_normalize import normalize_caption

log = logging.getLogger(__name__)

# Sentinel value used to mask padding tokens from the loss computation.
_LABEL_IGNORE_INDEX: int = -100


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AccessiCaptionDataset(Dataset):
    """PyTorch Dataset for WCAG-compliant image captioning.

    Each item returned is a dictionary with the following keys:

    ``pixel_values`` : torch.Tensor
        Image tensor after the processor's image transforms.
        Shape depends on the processor (typically ``[C, H, W]``).
    ``input_ids`` : torch.Tensor
        Token IDs of the WCAG-normalised caption.  Shape: ``[seq_len]``.
    ``attention_mask`` : torch.Tensor
        Attention mask (1 = real token, 0 = padding).  Shape: ``[seq_len]``.
    ``labels`` : torch.Tensor
        Copy of ``input_ids`` used as the language-modelling target.
        Padding positions are set to ``-100`` by the collator, not here.
        Shape: ``[seq_len]``.

    Parameters
    ----------
    data_path : str | Path
        Path to a processed JSON file (``train.json`` or ``eval.json``).
    processor : ProcessorMixin
        A HuggingFace ``AutoProcessor`` instance that handles both image
        pre-processing (``feature_extractor`` side) and text tokenisation
        (``tokenizer`` side).
    max_length : int, optional
        Maximum token length for captions.  Longer captions are truncated.
    skip_missing_images : bool, optional
        If ``True``, records whose ``image_path`` does not exist on disk are
        silently skipped.  If ``False`` (default) a ``FileNotFoundError`` is
        raised at index time.
    """

    def __init__(
        self,
        data_path: str | Path,
        processor: ProcessorMixin,
        max_length: int = 512,
        skip_missing_images: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.processor = processor
        self.max_length = max_length
        self.skip_missing_images = skip_missing_images

        self._records: list[dict[str, Any]] = self._load_records()

        # Cache the tokenizer reference for convenience
        self._tokenizer: PreTrainedTokenizerBase = getattr(processor, "tokenizer", processor)

        log.info(
            "AccessiCaptionDataset: loaded %d records from %s",
            len(self._records),
            self.data_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        """Read and return all records from the JSON data file.

        Returns
        -------
        list[dict[str, Any]]
            List of record dicts.

        Raises
        ------
        FileNotFoundError
            If *data_path* does not exist.
        ValueError
            If the file is not a valid JSON list.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        with self.data_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list in {self.data_path}, got {type(data).__name__}."
            )

        return data

    def _load_image(self, image_path: str) -> Image.Image:
        """Open an image from disk as an RGB PIL Image.

        Parameters
        ----------
        image_path:
            Filesystem path to the image file.

        Returns
        -------
        Image.Image
            RGB PIL image.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist and ``skip_missing_images`` is
            ``False``.
        """
        path = Path(image_path)
        if not path.exists():
            if self.skip_missing_images:
                log.debug("Image not found (using blank): %s", image_path)
                return Image.new("RGB", (224, 224), color=(128, 128, 128))
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            if self.skip_missing_images:
                log.warning("Could not open image %s (%s); using blank.", image_path, exc)
                return Image.new("RGB", (224, 224), color=(128, 128, 128))
            raise

    def _tokenize_caption(self, caption: str) -> dict[str, torch.Tensor]:
        """Tokenize a caption string and return ``input_ids`` + ``attention_mask``.

        WCAG normalisation is applied here so that any captions that bypassed
        the pipeline are still normalised at load time.

        Parameters
        ----------
        caption:
            Raw or already-normalised caption string.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"input_ids": Tensor[seq_len], "attention_mask": Tensor[seq_len]}``
        """
        normalised = normalize_caption(caption)
        encoding = self._tokenizer(
            normalised,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # [seq_len]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [seq_len]
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of records in this dataset split."""
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single dataset item.

        Parameters
        ----------
        idx:
            Index into the dataset.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: ``pixel_values``, ``input_ids``, ``attention_mask``, ``labels``.
        """
        record = self._records[idx]

        # --- Image ---
        image = self._load_image(record.get("image_path", ""))

        # Apply the processor's vision transforms; ``images=`` is the standard
        # HuggingFace kwarg for the image-feature-extractor side.
        image_encoding = self.processor(images=image, return_tensors="pt")
        # Shape: [1, C, H, W] → squeeze batch dim → [C, H, W]
        pixel_values: torch.Tensor = image_encoding["pixel_values"].squeeze(0)

        # --- Caption ---
        caption = record.get("caption", "")
        text_encoding = self._tokenize_caption(caption)

        input_ids: torch.Tensor = text_encoding["input_ids"]
        attention_mask: torch.Tensor = text_encoding["attention_mask"]

        # labels is a copy of input_ids; the collator will mask padding tokens
        labels: torch.Tensor = input_ids.clone()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------


@dataclass
class DataCollatorForVLMCaptioning:
    """Collator for batching ``AccessiCaptionDataset`` items.

    Pads ``input_ids``, ``attention_mask``, and ``labels`` to the length of
    the longest sequence in the batch.  Stacks ``pixel_values`` into a single
    tensor.  Label positions corresponding to padding tokens are set to
    ``-100`` so they are ignored by cross-entropy loss.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used to retrieve the ``pad_token_id``.
    padding_side : str, optional
        Side on which to add padding tokens; ``"right"`` (default) or
        ``"left"``.  Most decoder-only models expect right-padding during
        training.
    label_ignore_index : int, optional
        Value used to mask padding tokens in the labels tensor.
        Defaults to ``-100`` (the standard PyTorch cross-entropy ignore index).

    Examples
    --------
    >>> collator = DataCollatorForVLMCaptioning(tokenizer=tokenizer)
    >>> loader = DataLoader(dataset, batch_size=8, collate_fn=collator)
    """

    tokenizer: PreTrainedTokenizerBase
    padding_side: str = "right"
    label_ignore_index: int = field(default=_LABEL_IGNORE_INDEX)

    def __post_init__(self) -> None:
        """Validate that the tokenizer has a pad token."""
        if self.tokenizer.pad_token_id is None:
            # Fallback: use eos_token_id if pad_token_id is not set (GPT-style models)
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                log.warning(
                    "Tokenizer has no pad_token; using eos_token (%d) instead.",
                    self.tokenizer.pad_token_id,
                )
            else:
                raise ValueError(
                    "Tokenizer has neither pad_token_id nor eos_token_id. "
                    "Please set tokenizer.pad_token_id explicitly."
                )
        if self.padding_side not in {"left", "right"}:
            raise ValueError(
                f"padding_side must be 'left' or 'right', got {self.padding_side!r}."
            )

    # ------------------------------------------------------------------
    # Padding utilities
    # ------------------------------------------------------------------

    def _pad_sequence(
        self,
        sequences: list[torch.Tensor],
        pad_value: int,
    ) -> torch.Tensor:
        """Pad a list of 1-D tensors to the same length and stack them.

        Parameters
        ----------
        sequences:
            List of 1-D ``torch.Tensor`` objects with potentially different
            lengths.
        pad_value:
            Integer value used for padding tokens.

        Returns
        -------
        torch.Tensor
            2-D tensor of shape ``[batch_size, max_seq_len]``.
        """
        max_len = max(s.size(0) for s in sequences)
        padded: list[torch.Tensor] = []
        for seq in sequences:
            pad_len = max_len - seq.size(0)
            if pad_len == 0:
                padded.append(seq)
            elif self.padding_side == "right":
                padded.append(
                    torch.cat([seq, seq.new_full((pad_len,), fill_value=pad_value)])
                )
            else:  # left
                padded.append(
                    torch.cat([seq.new_full((pad_len,), fill_value=pad_value), seq])
                )
        return torch.stack(padded, dim=0)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(
        self, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collate a list of dataset items into a padded batch.

        Parameters
        ----------
        features:
            List of dicts, each with keys ``pixel_values``, ``input_ids``,
            ``attention_mask``, and ``labels``.

        Returns
        -------
        dict[str, torch.Tensor]
            Batch dict with the same keys.  Shapes:

            - ``pixel_values``   : ``[B, C, H, W]``
            - ``input_ids``      : ``[B, max_seq_len]``
            - ``attention_mask`` : ``[B, max_seq_len]``
            - ``labels``         : ``[B, max_seq_len]`` (padding → -100)
        """
        # --- Pixel values: stack (all images are already the same size) ---
        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)

        # --- Text sequences ---
        input_ids_list: list[torch.Tensor] = [f["input_ids"] for f in features]
        attention_mask_list: list[torch.Tensor] = [f["attention_mask"] for f in features]
        labels_list: list[torch.Tensor] = [f["labels"] for f in features]

        pad_id: int = self.tokenizer.pad_token_id  # type: ignore[assignment]

        input_ids = self._pad_sequence(input_ids_list, pad_value=pad_id)
        attention_mask = self._pad_sequence(attention_mask_list, pad_value=0)

        # Pad labels with the ignore index so loss doesn't penalise padding
        labels = self._pad_sequence(labels_list, pad_value=self.label_ignore_index)

        # Ensure label padding aligns with attention_mask: wherever the mask is
        # 0, the label must also be the ignore index (handles both left- and
        # right-padding edge cases).
        labels = labels.masked_fill(attention_mask == 0, self.label_ignore_index)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
