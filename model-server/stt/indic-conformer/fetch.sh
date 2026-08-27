#!/bin/bash
# Fetch this model's weights into its own folder.
#
# setup.sh runs <slot>/<model>/fetch.sh if it exists, so a model that needs
# weights brings its own download step and setup.sh needs to know nothing about
# it. Safe to re-run: it skips work already done.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="$HERE/models/IndicConformer.nemo"
URL="https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"

if [ -f "$TARGET" ]; then
  echo "  IndicConformer.nemo already present"
  exit 0
fi

mkdir -p "$HERE/models"
echo "  Downloading IndicConformer (~2.4 GB)"
wget -q --show-progress "$URL" -O "$TARGET"
