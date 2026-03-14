# Makefile — nf-core/airrflow BCR repertoire workflow
# Dataset: Galson et al. 2020 (PRJNA639956) — COVID-19 BCR-seq
#
# Usage:
#   make <target> [ENV=local|aws] [S3_BUCKET=your-bucket]
#
# Examples:
#   make data-galson2020
#   make run-airrflow-galson2020
#   ENV=aws S3_BUCKET=my-bucket make upload-galson2020
#   ENV=aws S3_BUCKET=my-bucket make run-airrflow-galson2020

ENV ?= local

.PHONY: help install-nextflow \
	data-galson2020 upload-galson2020 \
	run-airrflow-galson2020 \
	dry-run-galson2020 \
	analyse-galson2020 report-galson2020 build-report-image

help:
	@echo ""
	@echo "nf-core/airrflow — COVID-19 BCR Repertoire"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install-nextflow          Install Nextflow to ~/bin"
	@echo ""
	@echo "Data:"
	@echo "  make data-galson2020           Download Galson 2020 subset from SRA (~6 samples)"
	@echo "  make upload-galson2020         Upload FASTQs to S3 (requires S3_BUCKET)"
	@echo ""
	@echo "Pipeline:"
	@echo "  make run-airrflow-galson2020   Run nf-core/airrflow"
	@echo "  make dry-run-galson2020        Print pipeline command without running"
	@echo ""
	@echo "Analysis:"
	@echo "  make build-report-image        Build bcr-report container (immcantation + Quarto)"
	@echo "  make analyse-galson2020        Generate plots (R in container → analysis/plots/)"
	@echo "  make report-galson2020         Render Quarto report (→ report/galson2020.html)"
	@echo ""
	@echo "Options:"
	@echo "  ENV=aws S3_BUCKET=<bucket> make run-airrflow-galson2020"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────
install-nextflow:
	curl -s https://get.nextflow.io | bash
	mkdir -p ~/bin && mv nextflow ~/bin/
	@echo "Nextflow installed to ~/bin/nextflow"

# ── Data ───────────────────────────────────────────────────────
data-galson2020:
	bash data/galson2020/download.sh

upload-galson2020:
	./scripts/upload_to_s3.sh --dataset galson2020

# ── Pipeline ───────────────────────────────────────────────────
run-airrflow-galson2020:
	./scripts/run.sh --dataset galson2020 --env $(ENV)

dry-run-galson2020:
	./scripts/run.sh --dataset galson2020 --env $(ENV) --dry-run

# ── Analysis & Report ───────────────────────────────────────────
build-report-image:
	podman build -t bcr-report report/

analyse-galson2020:
	bash analysis/run.sh

report-galson2020:
	bash report/render.sh
