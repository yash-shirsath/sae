#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipe-path)
            PIPE_PATH="$2"
            shift 2
            ;;
        --session-name)
            SESSION_NAME="$2"
            shift 2
            ;;
        --sae-checkpoint)
            SAE_CHECKPOINT="$2"
            shift 2
            ;;
        --sae-act-dir)
            SAE_ACT_DIR="$2"
            shift 2
            ;;
        --generated-imgs-dir)
            GENERATED_IMGS_DIR="$2"
            shift 2
            ;;
        --artifact-bucket)
            ARTIFACT_BUCKET="$2"
            shift 2
            ;;
        --hookpoint)
            HOOKPOINT="$2"
            shift 2
            ;;
        --num-concepts)
            NUM_CONCEPTS="$2"
            shift 2
            ;;
        --prompts-per-concept)
            PROMPTS_PER_CONCEPT="$2"
            shift 2
            ;;
        --styles-per-prompt-gather)
            STYLES_PER_PROMPT_GATHER="$2"
            shift 2
            ;;
        --styles-per-prompt-generate)
            STYLES_PER_PROMPT_GENERATE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
required_params=(
    "PIPE_PATH"
    "SESSION_NAME"
    "SAE_CHECKPOINT"
    "SAE_ACT_DIR"
    "GENERATED_IMGS_DIR"
    "ARTIFACT_BUCKET"
    "HOOKPOINT"
    "NUM_CONCEPTS"
    "PROMPTS_PER_CONCEPT"
    "STYLES_PER_PROMPT_GATHER"
    "STYLES_PER_PROMPT_GENERATE"
    "STEPS"
)

for param in "${required_params[@]}"; do
    if [ -z "${!param}" ]; then
        echo "Error: Required parameter --${param,,} is not set"
        exit 1
    fi
done

# Generate a human-readable UUID name using adjectives and nouns
generate_readable_uuid() {
    adjectives=("swift" "brave" "calm" "eager" "fair" "gentle" "happy" "kind" "lively" "merry" "noble" "proud" "quiet" "sweet" "wise")
    nouns=("falcon" "eagle" "wolf" "bear" "lion" "tiger" "dolphin" "phoenix" "dragon" "hawk" "raven" "fox" "lynx" "panther" "shark")
    
    adj1=${adjectives[$RANDOM % ${#adjectives[@]}]}
    adj2=${adjectives[$RANDOM % ${#adjectives[@]}]}
    noun=${nouns[$RANDOM % ${#nouns[@]}]}
    
    echo "${adj1}-${adj2}-${noun}-$(date +%Y%m%d)"
}

# Create a unique run name
RUN_NAME=$(generate_readable_uuid)
echo "Generated run name: $RUN_NAME"

# Create config file with run parameters
cat > config.json << EOL
{
    "run_name": "$RUN_NAME",
    "pipe_path": "$PIPE_PATH",
    "session_name": "$SESSION_NAME",
    "sae_checkpoint": "$SAE_CHECKPOINT",
    "hookpoint": "$HOOKPOINT",
    "num_concepts": $NUM_CONCEPTS,
    "prompts_per_concept": $PROMPTS_PER_CONCEPT,
    "styles_per_prompt_gather": $STYLES_PER_PROMPT_GATHER,
    "styles_per_prompt_generate": $STYLES_PER_PROMPT_GENERATE,
    "steps": $STEPS
}
EOL

# Create the run directory in the artifact bucket
gsutil ls $ARTIFACT_BUCKET/$RUN_NAME/ 2>/dev/null || gsutil mkdir $ARTIFACT_BUCKET/$RUN_NAME/

# Upload config file
gsutil cp config.json $ARTIFACT_BUCKET/$RUN_NAME/

# Upload generated images
gsutil -m cp -r $GENERATED_IMGS_DIR/* $ARTIFACT_BUCKET/$RUN_NAME/$GENERATED_IMGS_DIR/

# Upload SAE activations
gsutil -m cp -r $SAE_ACT_DIR/* $ARTIFACT_BUCKET/$RUN_NAME/$SAE_ACT_DIR/

echo "Upload completed. Artifacts are available at: $ARTIFACT_BUCKET/$RUN_NAME/" 