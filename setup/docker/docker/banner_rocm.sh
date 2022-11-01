echo -e "\e[1;90m
==========================================
\e[m \e[1;91m
__      ___ _   _                   _____
\ \    / (_) | (_)            /\   |_   _|
 \ \  / / _| |_ _ ___ ______ /  \    | |
  \ \/ / | | __| / __|______/ /\ \   | |
   \  /  | | |_| \__ \     / ____ \ _| |_
    \/   |_|\__|_|___/    /_/    \_\_____|
\e[m \e[1;90m
==========================================
\e[m"
echo -e "Docker Image Version: \e[32m $VERSION \e[m"
echo -e "Vitis AI Git Hash: \e[32m $GIT_HASH \e[m"
echo "Build Date: $BUILD_DATE"
echo -e ""
echo -e "For PyTorch Workflows do:"
echo -e "    \e[31m conda activate vitis-ai-pytorch \e[m"
echo -e "For ROCm TensorFlow 2.3 Workflows do:"
echo -e "    \e[31m conda activate vitis-ai-rocm-tensorflow2.3 \e[m"
echo -e ""
