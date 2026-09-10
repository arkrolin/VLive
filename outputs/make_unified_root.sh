#!/bin/bash
# Flatten both corpora into one directory so the existing pipeline
# (build_static_pose.py uses root.iterdir(), i.e. ONE level only) works
# unchanged. Uses symlinks - no data is copied.
set -e
cd /root/work/nlp/xjzhao13/lijie_llama/VLive

rm -rf data/all
mkdir -p data/all

# ---- 1) standrad-live-2d (already flat) ----
n_std=0
for d in standrad-live-2d/*/; do
  name=$(basename "$d")
  ln -s "$(realpath "$d")" "data/all/std__${name}"
  n_std=$((n_std + 1))
done
echo "std  linked: $n_std"

# ---- 2) Live2d-model-master (nested: game/.../model) ----
# keep only dirs that have BOTH a .moc3 and at least one motion3.json
find Live2d-model-master -name '*.motion3.json' -printf '%h\n' 2>/dev/null | sort -u > /tmp/_motion_dirs.txt
n_l2=0
while IFS= read -r mdir; do
  mdir="${mdir%/motions}"
  [ -d "$mdir" ] || continue
  # the model dir itself or one level up must hold the .moc3
  if ! ls "$mdir"/*.moc3 >/dev/null 2>&1; then
    if ! ls "$mdir"/../*.moc3 >/dev/null 2>&1; then
      continue
    fi
    mdir="$(cd "$mdir/.." && pwd)"
  fi
  flat=$(printf '%s' "$mdir" | sed 's|^Live2d-model-master/||; s|/|__|g; s| |_|g; s|[()（）]|.|g')
  target="data/all/l2dm__${flat}"
  [ -e "$target" ] && continue
  ln -s "$(realpath "$mdir")" "$target"
  n_l2=$((n_l2 + 1))
done < /tmp/_motion_dirs.txt
echo "l2dm linked: $n_l2"

echo "total      : $(ls data/all | wc -l)"
echo "broken     : $(find data/all -xtype l 2>/dev/null | wc -l)"
