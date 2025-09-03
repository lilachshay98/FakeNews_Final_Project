## Git LFS Setup (required for large data files)

This repo stores large datasets with **Git LFS**. After cloning, run:

```bash
git lfs install
git lfs pull          # fetch the real blobs for all LFS-tracked files
git lfs checkout      # ensure working tree files are replaced (not pointer stubs)
