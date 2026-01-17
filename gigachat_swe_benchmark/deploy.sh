#!/bin/bash
#
# Deploy GigaChat SWE-bench benchmark to a new repository
#
# Usage:
#   ./deploy.sh <github-username> [repo-name] [--force]
#
# Example:
#   ./deploy.sh myusername gigachat-swe-benchmark
#   ./deploy.sh myusername gigachat-swe-benchmark --force  # overwrite existing remote
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
GITHUB_USER="${1:-}"
REPO_NAME="${2:-gigachat-swe-benchmark}"
FORCE_PUSH="${3:-}"

if [ -z "$GITHUB_USER" ]; then
    echo -e "${RED}Error: GitHub username required${NC}"
    echo "Usage: ./deploy.sh <github-username> [repo-name] [--force]"
    exit 1
fi

REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"
DEPLOY_DIR="/tmp/gigachat-deploy-$$"

echo "============================================================"
echo "  Deploying GigaChat SWE-bench Benchmark"
echo "============================================================"
echo "  GitHub User: $GITHUB_USER"
echo "  Repository:  $REPO_NAME"
echo "  URL:         $REPO_URL"
echo "============================================================"
echo ""

# Create deploy directory
echo -e "${YELLOW}[1/5]${NC} Creating deploy directory..."
mkdir -p "$DEPLOY_DIR"

# Copy project files (without model, results, __pycache__)
echo -e "${YELLOW}[2/5]${NC} Copying project files..."
cp "$SCRIPT_DIR/run_swebench_local.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/run_gigachat_swebench.sh" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/analyze_results.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/start_vllm_server.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/gigachat_swebench_local.yaml" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/gigachat_swebench.yaml" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/model_registry.json" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/README.md" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/deploy.sh" "$DEPLOY_DIR/"

# Create .gitignore
cat > "$DEPLOY_DIR/.gitignore" << 'EOF'
# Results
*_results/
*.traj.json
preds.json

# Python
__pycache__/
*.pyc
*.pyo
venv/
.venv/

# Model cache
*.safetensors
*.bin
*.pt

# Logs
*.log

# OS
.DS_Store
Thumbs.db
EOF

# Initialize git
echo -e "${YELLOW}[3/5]${NC} Initializing git repository..."
cd "$DEPLOY_DIR"
git init
git add .
git commit -m "Initial commit: GigaChat3-10B SWE-bench benchmark

- run_swebench_local.py: Local benchmark runner (no containers)
- run_gigachat_swebench.sh: Full pipeline script (requires Docker)
- analyze_results.py: Results analyzer with plots
- Configs for GigaChat model with vLLM"

# Add remote and push
echo -e "${YELLOW}[4/5]${NC} Adding remote origin..."
git remote add origin "$REPO_URL"
git branch -M main

echo -e "${YELLOW}[5/5]${NC} Pushing to GitHub..."
echo ""
echo -e "${YELLOW}NOTE: Make sure repository '$REPO_NAME' exists on GitHub!${NC}"
echo "Create it at: https://github.com/new"
echo ""
read -p "Press Enter when repository is created (or Ctrl+C to cancel)..."

if [ "$FORCE_PUSH" = "--force" ] || [ "$FORCE_PUSH" = "-f" ]; then
    echo -e "${YELLOW}Force pushing (will overwrite remote content)...${NC}"
    git push --force -u origin main
else
    git push -u origin main || {
        echo ""
        echo -e "${YELLOW}Push failed. If remote has existing content, run:${NC}"
        echo -e "  cd $DEPLOY_DIR"
        echo -e "  git push --force -u origin main"
        echo ""
        echo -e "Or re-run with --force flag:"
        echo -e "  ./deploy.sh $GITHUB_USER $REPO_NAME --force"
        exit 1
    }
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Deployment complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  Repository: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "  To clone on GPU machine:"
echo "    git clone $REPO_URL"
echo "    cd $REPO_NAME"
echo ""
echo "  Deploy directory: $DEPLOY_DIR"
echo "============================================================"
