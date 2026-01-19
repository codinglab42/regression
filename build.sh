#!/bin/bash
# build.sh - Script ottimizzato per Modern CMake + Pyenv

set -e  
set -o pipefail 

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   BUILD REGRESSION LIBRARY (PYENV)       ${NC}"
echo -e "${BLUE}==========================================${NC}"

# 1. RILEVAMENTO AMBIENTE (PYENV)
PYTHON_EXE=$(which python)
PYTHON_VER=$($PYTHON_EXE --version)

echo -e "${YELLOW}Using Python:${NC} $PYTHON_EXE ($PYTHON_VER)"

# 2. ARGOMENTI
CLEAN_BUILD="OFF"
for arg in "$@"; do
    [[ "$arg" == "--clean" ]] && CLEAN_BUILD="ON"
done

# 3. PULIZIA
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$CLEAN_BUILD" == "ON" ]]; then
    echo -e "${YELLOW}Pulizia directory build...${NC}"
    rm -rf "$PROJECT_ROOT/build"
fi

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"

# 4. CONFIGURAZIONE CMAKE
# Passiamo esplicitamente l'eseguibile di pyenv a CMake
echo -e "\n${BLUE}CONFIGURAZIONE CMAKE...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$PYTHON_EXE" \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 5. COMPILAZIONE
echo -e "\n${BLUE}COMPILAZIONE IN CORSO...${NC}"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
make -j$NPROC

# 6. VERIFICA RISULTATI
echo -e "\n${BLUE}CHECK GENERATED FILES...${NC}"

# Controlla librerie core
if ls lib/libregression* 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓ Librerie generate in build/lib:${NC}"
    ls -h lib/libregression*
else
    echo -e "${RED}✗ Errore: Librerie core non trovate!${NC}"
fi

# Controlla modulo python
echo -e "\n${YELLOW}Esecuzione test import Python...${NC}"
$PYTHON_EXE "$PROJECT_ROOT/tests/test_python_import.py"

# PY_MOD=$(find . -name "regression_module*.so")
# if [[ -n "$PY_MOD" ]]; then
#     echo -e "${GREEN}✓ Modulo Python generato:${NC} $PY_MOD"
#     # Test rapido di import
#     echo -n "Test import modulo... "
#     if $PYTHON_EXE -c "import sys; sys.path.insert(0, '$(dirname "$PY_MOD")'); import regression_module; print('OK')" 2>/dev/null; then
#         echo -e "${GREEN}SUCCESS${NC}"
#     else
#         echo -e "${RED}FAILED${NC} (Verifica i binding in C++)"
#     fi
# fi

echo -e "\n${BLUE}==========================================${NC}"
echo -e "${GREEN}BUILD COMPLETATA!${NC}"
echo -e "Esegui i test con: ${YELLOW}cd build && ctest${NC}"
echo -e "${BLUE}==========================================${NC}"