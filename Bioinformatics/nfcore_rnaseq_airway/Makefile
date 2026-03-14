# Makefile — nf-core airway RNA-seq + differential expression workflow
# Usage: make <target> [ENV=local|aws] [S3_BUCKET=your-bucket]
#
# Examples:
#   make workflow-airway                           # run both pipelines locally
#   ENV=aws S3_BUCKET=my-bucket make upload-airway
#   ENV=aws S3_BUCKET=my-bucket make workflow-airway

ENV ?= local

.PHONY: help install-nextflow \
	data-airway upload-airway \
	test-rnaseq-airway test-da-airway \
	workflow-airway \
	build-report-image analyse-airway report-airway

help:
	@echo ""
	@echo "nf-core Airway RNA-seq Workflow"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install-nextflow       Install Nextflow to ~/bin"
	@echo ""
	@echo "Data:"
	@echo "  make data-airway            Download airway dataset (GSE52778, ~3-4 GB)"
	@echo "  make upload-airway          Upload data to S3 (requires S3_BUCKET)"
	@echo ""
	@echo "Pipelines:"
	@echo "  make test-rnaseq-airway     Run nf-core/rnaseq on airway dataset"
	@echo "  make test-da-airway         Run nf-core/differentialabundance (requires rnaseq results)"
	@echo "  make workflow-airway        Run both pipelines end-to-end"
	@echo ""
	@echo "Analysis:"
	@echo "  make build-report-image     Build rnaseq-report container (R + Quarto)"
	@echo "  make analyse-airway         Generate plots (R in container -> analysis/plots/)"
	@echo "  make report-airway          Render Quarto report (-> report/airway.html)"
	@echo ""
	@echo "Options:"
	@echo "  ENV=aws S3_BUCKET=<bucket> make workflow-airway   Run on AWS Batch"
	@echo ""

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────
install-nextflow:
	curl -s https://get.nextflow.io | bash
	mkdir -p ~/bin && mv nextflow ~/bin/
	@echo "Nextflow installed to ~/bin/nextflow"

# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
data-airway:
	./data/airway/download.sh

upload-airway:
	./scripts/upload_to_s3.sh --dataset airway

# ──────────────────────────────────────────────────────────────
# Pipelines
# ──────────────────────────────────────────────────────────────
test-rnaseq-airway:
	./scripts/run.sh --pipeline rnaseq --dataset airway --env $(ENV)

test-da-airway:
	./scripts/run.sh --pipeline differentialabundance --dataset airway --env $(ENV)

# ──────────────────────────────────────────────────────────────
# Full workflow
# ──────────────────────────────────────────────────────────────
workflow-airway: test-rnaseq-airway test-da-airway

# ──────────────────────────────────────────────────────────────
# Analysis & Report
# ──────────────────────────────────────────────────────────────
build-report-image:
	podman build -t rnaseq-report report/

analyse-airway:
	bash analysis/run.sh

report-airway:
	bash report/render.sh
