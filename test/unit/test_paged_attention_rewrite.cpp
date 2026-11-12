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
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <limits>
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
  enum class Case {
    KeyCacheShape,
    ValueCacheRank,
    DynamicQueryStaticOutput,
    StaticRowMismatch,
    MetadataType,
    MetadataShape,
    MetadataTwo,
    MetadataThree,
    RotaryOffset,
    ValueHeadSize,
    NegativeScale,
    InfiniteScale,
    NanScale,
    ValidStatic
  };
  for (Case test : {Case::KeyCacheShape, Case::ValueCacheRank,
                    Case::DynamicQueryStaticOutput, Case::StaticRowMismatch,
                    Case::MetadataType, Case::MetadataShape, Case::MetadataTwo,
                    Case::MetadataThree, Case::RotaryOffset,
                    Case::ValueHeadSize, Case::NegativeScale,
                    Case::InfiniteScale, Case::NanScale, Case::ValidStatic}) {
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
    const char *description = nullptr;
    switch (test) {
    case Case::KeyCacheShape:
      description = "invalid key cache shape";
      attention->getResult(1).setType(
          RankedTensorType::get({2, 256, 1, 64}, Float16Type::get(&ctx)));
      break;
    case Case::ValueCacheRank:
      description = "unranked value cache";
      attention->getResult(2).setType(
          UnrankedTensorType::get(Float16Type::get(&ctx)));
      break;
    case Case::DynamicQueryStaticOutput:
      description = "dynamic query with static output rows";
      attention->getOperand(0).setType(RankedTensorType::get(
          {ShapedType::kDynamic, 192}, Float16Type::get(&ctx)));
      attention->getResult(0).setType(
          RankedTensorType::get({1, 64}, Float16Type::get(&ctx)));
      break;
    case Case::StaticRowMismatch:
      description = "mismatched static output rows";
      attention->getResult(0).setType(
          RankedTensorType::get({1, 64}, Float16Type::get(&ctx)));
      break;
    case Case::MetadataType:
    case Case::MetadataShape: {
      description = test == Case::MetadataType ? "invalid metadata type"
                                               : "invalid metadata shape";
      SmallVector<Value> extra(6, attention->getOperand(1));
      extra.push_back(
          attention->getOperand(test == Case::MetadataType ? 0 : 6));
      attention->insertOperands(10, extra);
      break;
    }
    case Case::ValidStatic:
      description = "matching static output rows";
      break;
    case Case::MetadataTwo:
    case Case::MetadataThree: {
      description = test == Case::MetadataTwo ? "two-element metadata"
                                              : "three-element metadata";
      auto function = cast<func::FuncOp>(attention->getParentOp());
      auto type = RankedTensorType::get({test == Case::MetadataTwo ? 2 : 3},
                                        IntegerType::get(&ctx, 32));
      Value metadata = function.front().addArgument(type, attention->getLoc());
      function.setType(FunctionType::get(&ctx,
                                         function.front().getArgumentTypes(),
                                         function.getResultTypes()));
      SmallVector<Value> extra(6, attention->getOperand(1));
      extra.push_back(metadata);
      attention->insertOperands(10, extra);
      break;
    }
    case Case::RotaryOffset:
    case Case::ValueHeadSize:
      description = test == Case::RotaryOffset ? "unsupported rotary offset"
                                               : "unsupported value head size";
      attention->setAttr(test == Case::RotaryOffset ? "rotary_offset"
                                                    : "v_head_size",
                         IntegerAttr::get(IntegerType::get(&ctx, 64), 8));
      break;
    case Case::NegativeScale:
    case Case::InfiniteScale:
    case Case::NanScale:
      description = test == Case::NegativeScale   ? "negative scale"
                    : test == Case::InfiniteScale ? "infinite scale"
                                                  : "NaN scale";
      attention->setAttr(
          "scale",
          FloatAttr::get(Float32Type::get(&ctx),
                         test == Case::NegativeScale ? -1.0f
                         : test == Case::InfiniteScale
                             ? std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::quiet_NaN()));
      break;
    }
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
    const bool expectRejection = test != Case::ValidStatic &&
                                 test != Case::MetadataTwo &&
                                 test != Case::MetadataThree;
    const bool passed = expectRejection ? rejected && errors == 1 && unchanged
                                        : !rejected && errors == 0 &&
                                              succeeded(verify(*module));
    std::printf("%s %s: %s\n", passed ? "PASS" : "FAIL", description,
                expectRejection ? "rejected without IR mutation" : "accepted");
    failures += !passed;
  }
  return failures ? 1 : 0;
}
