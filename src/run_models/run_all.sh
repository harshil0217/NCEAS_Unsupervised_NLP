#!/bin/bash

# Define the themes and parameter sets directly in the script
#"Energy, Ecosystems, and Humans" 
themes=("Energy, Ecosystems, and Humans" "Offshore energy impacts on fisheries")
# parameter_sets=(
#   '{"t": 1.0, "max_sub": 3, "depth": 5, "synonyms": 0, "branching": "random","add_noise":.25}'
#   '{"t": 1.0, "max_sub": 5, "depth": 3, "synonyms": 0, "branching": "random","add_noise":.25}'
#   '{"t": 1.0, "max_sub": 5, "depth": 3, "synonyms": 10, "branching": "random","add_noise":.25}'
#   '{"t": 1.0, "max_sub": 3, "depth": 5, "synonyms": 10, "branching": "random","add_noise":.25}'
#   '{"t": 1.0, "max_sub": 5, "depth": 3, "synonyms": 0, "branching": "random","add_noise":0}'
#   '{"t": 1.0, "max_sub": 3, "depth": 5, "synonyms": 0, "branching": "random","add_noise":0}'

# )
parameter_sets=(
  '{"t": 1.0, "max_sub": 5, "depth": 3, "synonyms": 0, "branching": "random","add_noise":0}'
  #'{"t": 1.0, "max_sub": 3, "depth": 5, "synonyms": 0, "branching": "random","add_noise":0}'
)
# re run noise??? make sure no extra set is generated. delete old maybe?

# Loop through each theme
for theme in "${themes[@]}"; do
  echo "Running for theme: $theme"
  
  # Loop through each parameter set
  for param_set in "${parameter_sets[@]}"; do
    t=$(echo "$param_set" | jq -r '.t')
    max_sub=$(echo "$param_set" | jq -r '.max_sub')
    depth=$(echo "$param_set" | jq -r '.depth')
    synonyms=$(echo "$param_set" | jq -r '.synonyms')
    branching=$(echo "$param_set" | jq -r '.branching')
    add_noise=$(echo "$param_set" | jq -r '.add_noise')
        

    echo "Generating with params -> t: $t, max_sub: $max_sub, depth: $depth, synonyms: $synonyms, branching: $branching, add_noise: $add_noise"
    python "data_generation/generate.py" --theme "$theme" --t "$t" --max_sub "$max_sub" --depth "$depth" --synonyms "$synonyms" --branching "$branching" --add_noise  "$add_noise"



    # Print the extracted parameters for verification
    echo "Running with params -> t: $t, max_sub: $max_sub, depth: $depth, synonyms: $synonyms, branching: $branching"
    
    # Run with wait=True
    echo "Running with wait=False..."
    python run_models/eval_script.py --theme "$theme" --t "$t" --max_sub "$max_sub" --depth "$depth" --synonyms "$synonyms" --branching "$branching" --add_noise  "$add_noise" --wait False
    
    # Run with wait=False
    echo "Running with wait=True..."
    python run_models/eval_script.py --theme "$theme" --t "$t" --max_sub "$max_sub" --depth "$depth" --synonyms "$synonyms" --branching "$branching" --add_noise  "$add_noise" --wait True
  done
done
