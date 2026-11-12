/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "OnnxToHipUtils.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

using namespace mlir;

int main() {
  MLIRContext ctx;
  ctx.allowUnregisteredDialects();
  ctx.loadDialect<hip::HipDialect, arith::ArithDialect, func::FuncDialect,
                  tensor::TensorDialect>();
  RewritePatternSet patterns(&ctx);
  hip::populatePagedAttentionConversionPatterns(patterns, &ctx);
  int failures = 0;
  for (unsigned index : {1u, 2u}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      func.func @main_graph(%ctx: !hip.context, %query: tensor<2x192xf16>,
                           %cache: tensor<1x256x1x64xf16>,
                           %cu: tensor<2xi32>, %past: tensor<1xi32>,
                           %table: tensor<1x1xi32>, %rope: tensor<256x32xf16>)
          -> (tensor<2x64xf16>, tensor<1x256x1x64xf16>, tensor<1x256x1x64xf16>) {
        %none = "onnx.NoValue"() {value} : () -> none
        %result:3 = "onnx.Custom"(%query, %none, %none, %cache, %cache,
                                 %cu, %past, %table, %rope, %rope)
            {function_name = "PagedAttention", domain_name = "com.microsoft",
             num_heads = 1 : si64, kv_num_heads = 1 : si64, do_rotary = 1 : si64}
            : (tensor<2x192xf16>, none, none, tensor<1x256x1x64xf16>,
               tensor<1x256x1x64xf16>, tensor<2xi32>, tensor<1xi32>,
               tensor<1x1xi32>, tensor<256x32xf16>, tensor<256x32xf16>)
            -> (tensor<2x64xf16>, tensor<1x256x1x64xf16>, tensor<1x256x1x64xf16>)
        return %result#0, %result#1, %result#2 : tensor<2x64xf16>,
            tensor<1x256x1x64xf16>, tensor<1x256x1x64xf16>
      }
    )mlir",
                                              &ctx);
    if (!module)
      return 1;
    Operation *attention = nullptr;
    module->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "onnx.Custom")
        attention = op;
    });
    if (!attention)
      return 1;
    attention->getResult(index).setType(
        index == 1 ? Type(RankedTensorType::get({2, 256, 1, 64},
                                                Float16Type::get(&ctx)))
                   : Type(UnrankedTensorType::get(Float16Type::get(&ctx))));
    auto snapshot = [&] {
      std::string text;
      llvm::raw_string_ostream stream(text);
      module->print(stream);
      return text;
    };
    const std::string before = snapshot();
    unsigned errors = 0;
    ScopedDiagnosticHandler diagnostics(&ctx, [&](Diagnostic &) {
      ++errors;
      return success();
    });
    PatternRewriter rewriter(&ctx);
    rewriter.setInsertionPoint(attention);
    const bool rejected =
        failed(patterns.getNativePatterns().front()->matchAndRewrite(attention,
                                                                     rewriter));
    const bool unchanged = snapshot() == before;
    const bool passed = rejected && errors == 1 && unchanged;
    std::printf("%s invalid cache result %u leaves IR unchanged\n",
                passed ? "PASS" : "FAIL", index);
    failures += !passed;
  }
  return failures ? 1 : 0;
}
