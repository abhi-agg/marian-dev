#!/bin/bash

if [[ ! -e ../models ]]; then
    mkdir -p ../models
    if [[ ! -e ../students ]]; then
        echo "Cloning https://github.com/browsermt/students)"
        git clone --depth 1 --branch main --single-branch https://github.com/browsermt/students ../
    fi
    
    echo "Downloading files"
    ../students/esen/download-models.sh

    echo "Copying downloaded files to models folder"
    cp ../students/esen/esen.student.tiny11/vocab.esen* ../students/esen/esen.student.tiny11/model* ../students/esen/esen.student.tiny11/lex.s2t* ../models/
    sacrebleu -t wmt13 -l es-en --echo src > ../models/newstest2013.es
    head -n300 ../models/newstest2013.es > ../models/newstest2013.es.top300lines
    gunzip ../models/*
else
    echo "models directory already exists in root folder; Using it to package files without downloading anything"
fi

echo "Packaging files for wasm binary"
$EMSDK_PYTHON $EMSDK/upstream/emscripten/tools/file_packager.py model-files.data --preload ../models/@ --js-output=model-files.js

echo "Enabling wormhole via APIs that compile and instantiate wasm module"
sed -i.bak 's/var result = WebAssembly.instantiateStreaming(response, info);/var result = WebAssembly.instantiateStreaming(response, info, {simdWormhole:true});/g' marian-decoder.js
sed -i.bak 's/return WebAssembly.instantiate(binary, info);/return WebAssembly.instantiate(binary, info, {simdWormhole:true});/g' marian-decoder.js
sed -i.bak 's/var module = new WebAssembly.Module(bytes);/var module = new WebAssembly.Module(bytes, {simdWormhole:true});/g' marian-decoder.js
echo "SUCCESS"

echo "Polyfill the fallback integer (8-bit) gemm implementation from the main module"
sed -i.bak 's/asmLibraryArg,/asmLibraryArg,"wasm_gemm":{\
    "int8_prepare_a": (...a) => Module["asm"].int8PrepareAFallback(...a),\
    "int8_prepare_b": (...a) => Module["asm"].int8PrepareBFallback(...a),\
    "int8_prepare_b_from_transposed": (...a) => Module["asm"].int8PrepareBFromTransposedFallback(...a),\
    "int8_prepare_b_from_quantized_transposed": (...a) => Module["asm"].int8PrepareBFromQuantizedTransposedFallback(...a),\
    "int8_prepare_bias": (...a) => Module["asm"].int8PrepareBiasFallback(...a),\
    "int8_multiply_and_add_bias": (...a) => Module["asm"].int8MultiplyAndAddBiasFallback(...a),\
    "int8_select_columns_of_b": (...a) => Module["asm"].int8SelectColumnsOfBFallback(...a),\
    },/g' marian-decoder.js
echo "SUCCESS"
