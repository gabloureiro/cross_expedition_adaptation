# Images

This directory is intentionally empty.

The photographic materials used in this paper are publicly available on
the DeepData platform of the International Seabed Authority (ISA)

Once obtained, place the tiled images in the following structure:
  images/labelled/     —  labelled ground-truth tiles (640×640 px)
  images/unlabelled/   —  unlabelled target tiles (640×640 px)

Tile filenames follow the pattern: {year}_tile{N}.jpeg
Negative samples (background tiles without nodules) use the prefix: unknown_tile{N}.jpeg
