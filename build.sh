#!/bin/bash
# build_all.sh

echo "=========================================="
echo "BUILD LIBRERIE STATICHE, DINAMICHE E PYTHON"
echo "=========================================="

# Directory progetto
PROJECT_ROOT="$(pwd)"
echo "Directory: $PROJECT_ROOT"

# Pulisci tutto
echo ""
echo "1. PULIZIA COMPLETA..."
rm -rf "$PROJECT_ROOT/build"
rm -rf "$PROJECT_ROOT/lib"
mkdir -p "$PROJECT_ROOT/build"
mkdir -p "$PROJECT_ROOT/lib"

# Configura CMake
echo ""
echo "2. CONFIGURAZIONE CMAKE..."
cd "$PROJECT_ROOT/build"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compila
echo ""
echo "3. COMPILAZIONE..."
make -j$(nproc)

echo ""
echo "4. VERIFICA LIBRERIE CREATE..."
echo "   In build/lib/:"
ls -la "$PROJECT_ROOT/build/lib/" 2>/dev/null || echo "   Nessun file"

echo ""
echo "   In progetto lib/:"
ls -la "$PROJECT_ROOT/lib/" 2>/dev/null || echo "   Nessun file"

echo ""
echo "5. DETTAGLI LIBRERIE:"
echo "   Librerie statiche (.a):"
find "$PROJECT_ROOT" -name "*.a" -type f | xargs ls -la 2>/dev/null || echo "   Nessuna"
echo ""
echo "   Librerie dinamiche (.so):"
find "$PROJECT_ROOT" -name "*.so" -type f | xargs ls -la 2>/dev/null || echo "   Nessuna"

echo ""
echo "=========================================="
echo "BUILD COMPLETATO!"
echo "=========================================="