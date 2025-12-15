###############################
# KONFIGURATION
###############################

# Gesamtzahl der Iterationen
ITERATIONS=10000

# Checkpoints (können beliebig viele sein)
CHECKPOINTS=(1000)

# Bild-Resolution (1 = Original, 2 = Halb, 4 = Viertel)
RESOLUTION=1

# Ob finale Evaluation (metrics.py) laufen soll: true/false
FINAL_EVAL=true


SOURCE_PATH="/home/bennet/garden"

# Prozentsatz der zufällig ausgewählten SH-Koeffizienten
PERCENTAGE=20
EVERY=200


cd ~/gaussian-splatting
   #mkdir -p "${MODEL_DIR}"

    python train.py \
      -s "${SOURCE_PATH}" \
      --eval \
      --data_device cuda \
      --adaptive_sh \
      --visualize_degrees \
      --visualize_gradients \
      --visualize_gradients_iters 2500 4500 7500\
      --resolution "${RESOLUTION}" \
      --iterations "${ITERATIONS}" \
      #--sh_percentage "${PERCENTAGE}" "${EVERY}" \

    echo "Training abgeschlossen."

    #--visualize_degrees \
    #--visualize_gradients_iters 1000\
    #--adaptive_sh \