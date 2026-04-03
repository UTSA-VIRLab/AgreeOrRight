#!/bin/bash
# =============================================================
# MIMIC-CXR-JPG v2.1.0 Download Script
# =============================================================
# PREREQUISITES:
# 1. Complete CITI "Data or Specimens Only Research" course
# 2. Sign MIMIC-CXR DUA on PhysioNet
# 3. Configure GCP: gcloud auth login && gcloud config set project YOUR_PROJECT
# 4. Accept the dataset agreement on PhysioNet
# =============================================================

set -euo pipefail

DEST_DIR="/raid/den365/AgenticMedXAI_CVPR2026/data/benchmarks/mimic_cxr"
GCS_BUCKET="gs://mimic-cxr-jpg-2.1.0.physionet.org/"
LOG_FILE="/raid/den365/AgenticMedXAI_CVPR2026/outputs/logs/mimic_download.log"

echo "============================================="
echo "MIMIC-CXR-JPG v2.1.0 Download"
echo "============================================="
echo "Destination: ${DEST_DIR}"
echo "Source: ${GCS_BUCKET}"
echo ""

# Check prerequisites
if ! command -v gsutil &>/dev/null; then
    echo "ERROR: gsutil not found. Install Google Cloud SDK first."
    echo "  curl https://sdk.cloud.google.com | bash"
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1; then
    echo "ERROR: No active GCP authentication found."
    echo "  Run: gcloud auth login"
    exit 1
fi

# Create destination directory
mkdir -p "${DEST_DIR}"

echo "Starting high-throughput transfer..."
echo "Logging to: ${LOG_FILE}"

# High-throughput parallel download
gsutil -m cp -r "${GCS_BUCKET}" "${DEST_DIR}/" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "Download complete. Verifying integrity..."

# Verify SHA256 checksums
if [ -f "${DEST_DIR}/SHA256SUMS.txt" ]; then
    echo "Running SHA256 verification (this may take a while)..."
    cd "${DEST_DIR}"
    sha256sum -c SHA256SUMS.txt 2>&1 | tee -a "${LOG_FILE}"
    echo "Integrity verification complete."
else
    echo "WARNING: SHA256SUMS.txt not found. Manual verification recommended."
fi

# Print summary
echo ""
echo "============================================="
echo "Download Summary"
echo "============================================="
echo "Location: ${DEST_DIR}"
du -sh "${DEST_DIR}" 2>/dev/null || echo "Size calculation failed"
find "${DEST_DIR}" -name "*.jpg" 2>/dev/null | wc -l | xargs -I{} echo "JPEG files: {}"
echo "============================================="
