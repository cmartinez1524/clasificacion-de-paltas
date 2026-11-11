"""Dataset, aumentaciones y dataloaders.

Decision de diseno clave: las aumentaciones de color son deliberadamente suaves.
En la mayoria de las tareas de clasificacion el color es una variable molesta y
conviene perturbarlo con fuerza; aca el color ES la senal -- la transicion de
verde a cafe oscuro es literalmente la definicion del indice de madurez. Un
ColorJitter agresivo (el tipico brightness=0.4, saturation=0.4, hue=0.1)
convertiria una clase 2 en una clase 4 sin cambiarle la etiqueta, inyectando
ruido de etiquetado. Perturbamos brillo y contraste lo justo para cubrir la
variacion de iluminacion esperable, y el tono casi nada.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Orden fijo de los grupos de almacenamiento para el one-hot de metadatos.
STORAGE_GROUPS = ("T10", "T20", "Tam")
#: Dia maximo observado en el dataset (T10); se usa para escalar el dia a [0, 1].
DAY_MAX = 26.0
#: Dimension del vector de metadatos: one-hot del grupo + dia escalado.
N_META = len(STORAGE_GROUPS) + 1


def meta_vector(df: pd.DataFrame) -> torch.Tensor:
    """Codifica temperatura de almacenamiento y dia como un vector de 4 valores.

    Es exactamente la informacion que usa la referencia sin imagen
    (scripts/10_referencia_metadatos.py); el modelo con metadatos la recibe
    junto con la foto para que la comparacion sea directa.
    """
    onehot = torch.tensor(
        [[float(g == s) for s in STORAGE_GROUPS] for g in df["storage_group"]],
        dtype=torch.float32,
    )
    dia = torch.tensor(df["day"].to_numpy() / DAY_MAX, dtype=torch.float32).unsqueeze(1)
    return torch.cat([onehot, dia], dim=1)


class AvocadoDataset(Dataset):
    """Fotografias de paltas Hass etiquetadas con el indice de madurez 1-5.

    Las etiquetas se devuelven en el rango 0-4 (indice de clase de PyTorch);
    la conversion a 1-5 se hace solo al reportar.
    """

    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None,
                 with_meta: bool = False):
        self.paths = [Path(images_dir) / f for f in df["image_file"]]
        self.labels = torch.tensor(df["ripening_index"].to_numpy() - 1, dtype=torch.long)
        self.sample_ids = df["sample_id"].to_numpy()
        self.storage_groups = df["storage_group"].to_numpy()
        self.file_names = df["file_name"].to_numpy()
        self.transform = transform
        self.meta = meta_vector(df) if with_meta else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.meta is not None:
            return img, self.meta[i], self.labels[i]
        return img, self.labels[i]


def build_transforms(img_size: int = 224, train: bool = False):
    if not train:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        # La fruta esta siempre centrada y ocupa un area similar; recortes muy
        # agresivos la sacarian de cuadro, asi que acotamos scale por abajo.
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        # La orientacion del pedunculo varia entre tomas: rotar es realista.
        transforms.RandomRotation(20, fill=(255, 255, 255)),
        # Suave a proposito: ver docstring del modulo.
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random"),
    ])


def build_dataloaders(
    splits_csv: Path,
    images_dir: Path,
    img_size: int = 224,
    batch_size: int = 48,
    num_workers: int = 8,
    seed: int = 42,
    with_meta: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    """Crea los tres dataloaders y devuelve tambien los DataFrames por particion."""
    df = pd.read_csv(splits_csv)
    loaders: dict[str, DataLoader] = {}
    frames: dict[str, pd.DataFrame] = {}

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        frames[split] = sub
        is_train = split == "train"
        ds = AvocadoDataset(sub, images_dir, build_transforms(img_size, train=is_train),
                           with_meta=with_meta)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
    return loaders, frames
