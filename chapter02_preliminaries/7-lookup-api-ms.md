# 查阅文档


由于篇幅限制，本书不可能介绍每一个PyTorch函数和类。
API文档、其他教程和示例提供了本书之外的大量文档。
本节提供了一些查看PyTorch API的指导。


## 查找模块中的所有函数和类

为了知道模块中可以调用哪些函数和类，可以调用`dir`函数。
例如，我们可以(**查询ops运算中的所有属性：**)



```python
import mindspore as ms

print(dir(ms.ops))
```

    ['ACos', 'Abs', 'AccumulateNV2', 'Acosh', 'Adam', 'AdamNoUpdateParam', 'AdamWeightDecay', 'AdaptiveAvgPool2D', 'AdaptiveAvgPool3D', 'AdaptiveMaxPool2D', 'AdaptiveMaxPool3D', 'Add', 'AddN', 'Addcdiv', 'Addcmul', 'AdjustHue', 'AdjustSaturation', 'AffineGrid', 'AiCPURegOp', 'AkgAscendRegOp', 'AkgGpuRegOp', 'AllGather', 'AllReduce', 'AlltoAll', 'Angle', 'ApplyAdaMax', 'ApplyAdadelta', 'ApplyAdagrad', 'ApplyAdagradDA', 'ApplyAdagradV2', 'ApplyAdamWithAmsgrad', 'ApplyAddSign', 'ApplyCenteredRMSProp', 'ApplyFtrl', 'ApplyGradientDescent', 'ApplyKerasMomentum', 'ApplyMomentum', 'ApplyPowerSign', 'ApplyProximalAdagrad', 'ApplyProximalGradientDescent', 'ApplyRMSProp', 'ApproximateEqual', 'ArgMaxWithValue', 'ArgMinWithValue', 'Argmax', 'Argmin', 'Asin', 'Asinh', 'Assert', 'Assign', 'AssignAdd', 'AssignSub', 'Atan', 'Atan2', 'Atanh', 'AvgPool', 'AvgPool3D', 'BCEWithLogitsLoss', 'BNTrainingReduce', 'BNTrainingUpdate', 'BartlettWindow', 'BasicLSTMCell', 'BatchMatMul', 'BatchNorm', 'BatchToSpace', 'BatchToSpaceND', 'BatchToSpaceNDV2', 'Bernoulli', 'BesselI0', 'BesselI0e', 'BesselI1', 'BesselI1e', 'BesselJ0', 'BesselJ1', 'BesselK0', 'BesselK0e', 'BesselK1', 'BesselK1e', 'BesselY0', 'BesselY1', 'Betainc', 'BiasAdd', 'BinaryCrossEntropy', 'Bincount', 'BitwiseAnd', 'BitwiseOr', 'BitwiseXor', 'BlackmanWindow', 'BoundingBoxDecode', 'BoundingBoxEncode', 'Broadcast', 'BroadcastTo', 'Bucketize', 'BufferAppend', 'BufferGetItem', 'BufferSample', 'CTCGreedyDecoder', 'CTCLoss', 'CTCLossV2', 'Cast', 'Cauchy', 'Cdist', 'CeLU', 'Ceil', 'ChannelShuffle', 'CheckNumerics', 'CheckValid', 'Cholesky', 'CholeskyInverse', 'CholeskySolve', 'Coalesce', 'Col2Im', 'CombinedNonMaxSuppression', 'CompareAndBitpack', 'Complex', 'ComplexAbs', 'ComputeAccidentalHits', 'Concat', 'ConfusionMatrix', 'Conj', 'ConjugateTranspose', 'Conv2D', 'Conv2DBackpropInput', 'Conv2DTranspose', 'Conv3D', 'Conv3DTranspose', 'Cos', 'Cosh', 'CountNonZero', 'CpuRegOp', 'CropAndResize', 'Cross', 'CumProd', 'CumSum', 'Cummax', 'Cummin', 'CumulativeLogsumexp', 'Custom', 'CustomRegOp', 'DType', 'DataFormatDimMap', 'DataFormatVecPermute', 'DataType', 'DeformableOffsets', 'Depend', 'DepthToSpace', 'DepthwiseConv2dNative', 'Diag', 'DiagPart', 'Digamma', 'Dilation2D', 'Div', 'DivNoNan', 'Dropout', 'Dropout2D', 'Dropout3D', 'DropoutDoMask', 'DropoutGenMask', 'DynamicGRUV2', 'DynamicRNN', 'DynamicShape', 'EditDistance', 'Eig', 'Einsum', 'Elu', 'EmbeddingLookup', 'Eps', 'Equal', 'EqualCount', 'Erf', 'Erfc', 'Erfinv', 'EuclideanNorm', 'Exp', 'Expand', 'ExpandDims', 'Expm1', 'ExtractGlimpse', 'ExtractImagePatches', 'ExtractVolumePatches', 'Eye', 'FFTWithSize', 'FastGeLU', 'FastGelu', 'Fill', 'FillDiagonal', 'FillV2', 'Fills', 'Flatten', 'FloatStatus', 'Floor', 'FloorDiv', 'FloorMod', 'Fmax', 'Fmin', 'FractionalAvgPool', 'FractionalMaxPool', 'FractionalMaxPool3DWithFixedKsize', 'FractionalMaxPoolWithFixedKsize', 'FusedAdaFactor', 'FusedAdaFactorWithGlobalNorm', 'FusedCastAdamWeightDecay', 'FusedSparseAdam', 'FusedSparseFtrl', 'FusedSparseLazyAdam', 'FusedSparseProximalAdagrad', 'FusedWeightScaleApplyMomentum', 'GLU', 'Gamma', 'Gather', 'GatherD', 'GatherNd', 'GatherV2', 'Gcd', 'GeLU', 'GeSwitch', 'Gelu', 'Geqrf', 'Ger', 'GetNext', 'GradOperation', 'Greater', 'GreaterEqual', 'GridSampler2D', 'GridSampler3D', 'HSVToRGB', 'HShrink', 'HSigmoid', 'HSwish', 'HammingWindow', 'Heaviside', 'Histogram', 'HistogramFixedWidth', 'HistogramSummary', 'HookBackward', 'HyperMap', 'Hypot', 'IOU', 'Identity', 'IdentityN', 'Igamma', 'Igammac', 'Im2Col', 'Imag', 'ImageSummary', 'InTopK', 'IndexAdd', 'IndexFill', 'IndexPut', 'InplaceAdd', 'InplaceIndexAdd', 'InplaceSub', 'InplaceUpdate', 'InplaceUpdateV2', 'InsertGradientOf', 'Inv', 'Invert', 'InvertPermutation', 'IsClose', 'IsFinite', 'IsInf', 'IsNan', 'J', 'KLDivLoss', 'L2Loss', 'L2Normalize', 'LARSUpdate', 'LRN', 'LSTM', 'LayerNorm', 'Lcm', 'LeftShift', 'Lerp', 'Less', 'LessEqual', 'Lgamma', 'LinSpace', 'ListDiff', 'Log', 'Log1p', 'LogMatrixDeterminant', 'LogNormalReverse', 'LogSoftmax', 'LogSpace', 'LogUniformCandidateSampler', 'LogicalAnd', 'LogicalNot', 'LogicalOr', 'LogicalXor', 'Logit', 'LowerBound', 'LpNorm', 'Lstsq', 'LuSolve', 'LuUnpack', 'Map', 'MapCacheIdx', 'MapUniform', 'MaskedFill', 'MaskedScatter', 'MaskedSelect', 'MatMul', 'MatrixBandPart', 'MatrixDeterminant', 'MatrixDiagPartV3', 'MatrixDiagV3', 'MatrixExp', 'MatrixInverse', 'MatrixLogarithm', 'MatrixPower', 'MatrixSetDiagV3', 'MatrixSolve', 'MatrixSolveLs', 'MatrixTriangularSolve', 'MaxPool', 'MaxPool3D', 'MaxPool3DWithArgmax', 'MaxPoolWithArgmax', 'MaxPoolWithArgmaxV2', 'MaxUnpool2D', 'MaxUnpool3D', 'Maximum', 'Median', 'Merge', 'Meshgrid', 'Minimum', 'MirrorPad', 'Mish', 'Mod', 'Mul', 'MulNoNan', 'MultiMarginLoss', 'MultilabelMarginLoss', 'Multinomial', 'MultinomialWithReplacement', 'MultitypeFuncGraph', 'Mvlgamma', 'NLLLoss', 'NMSWithMask', 'NPUAllocFloatStatus', 'NPUClearFloatStatus', 'NPUGetFloatStatus', 'NanToNum', 'Neg', 'NeighborExchange', 'NeighborExchangeV2', 'NextAfter', 'NoRepeatNGram', 'NonDeterministicInts', 'NonMaxSuppressionV3', 'NonMaxSuppressionWithOverlaps', 'NonZero', 'NotEqual', 'NthElement', 'NuclearNorm', 'OneHot', 'Ones', 'OnesLike', 'Orgqr', 'Ormqr', 'P', 'PReLU', 'Pack', 'Pad', 'PadV3', 'Padding', 'ParallelConcat', 'ParameterizedTruncatedNormal', 'Partial', 'Pdist', 'Poisson', 'Polar', 'Polygamma', 'PopulationCount', 'Pow', 'Primitive', 'PrimitiveWithCheck', 'PrimitiveWithInfer', 'Print', 'Pull', 'Push', 'PyExecute', 'PyFunc', 'Qr', 'Quantile', 'RGBToHSV', 'RNNTLoss', 'ROIAlign', 'RaggedRange', 'RandomCategorical', 'RandomChoiceWithMask', 'RandomGamma', 'RandomPoisson', 'RandomShuffle', 'Randperm', 'RandpermV2', 'Range', 'Rank', 'ReLU', 'ReLU6', 'ReLUV2', 'Real', 'RealDiv', 'Reciprocal', 'ReduceAll', 'ReduceAny', 'ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceOp', 'ReduceProd', 'ReduceScatter', 'ReduceStd', 'ReduceSum', 'Renorm', 'Reshape', 'ResizeArea', 'ResizeBicubic', 'ResizeBilinear', 'ResizeBilinearV2', 'ResizeLinear1D', 'ResizeNearestNeighbor', 'ResizeNearestNeighborV2', 'ReverseSequence', 'ReverseV2', 'RightShift', 'Rint', 'Roll', 'Round', 'Rsqrt', 'SGD', 'STFT', 'SampleDistortedBoundingBoxV2', 'ScalarCast', 'ScalarSummary', 'ScalarToArray', 'ScalarToTensor', 'ScaleAndTranslate', 'ScatterAdd', 'ScatterAddWithAxis', 'ScatterDiv', 'ScatterMax', 'ScatterMin', 'ScatterMul', 'ScatterNd', 'ScatterNdAdd', 'ScatterNdDiv', 'ScatterNdMax', 'ScatterNdMin', 'ScatterNdMul', 'ScatterNdSub', 'ScatterNdUpdate', 'ScatterNonAliasingAdd', 'ScatterSub', 'ScatterUpdate', 'SeLU', 'SearchSorted', 'SegmentMax', 'SegmentMean', 'SegmentMin', 'SegmentProd', 'SegmentSum', 'Select', 'Shape', 'Sigmoid', 'SigmoidCrossEntropyWithLogits', 'Sign', 'Sin', 'Sinc', 'Sinh', 'Size', 'Slice', 'SliceGetItem', 'SmoothL1Loss', 'SoftMarginLoss', 'SoftShrink', 'Softmax', 'SoftmaxCrossEntropyWithLogits', 'Softplus', 'Softsign', 'Sort', 'SpaceToBatch', 'SpaceToBatchND', 'SpaceToDepth', 'SparseApplyAdadelta', 'SparseApplyAdagrad', 'SparseApplyAdagradV2', 'SparseApplyFtrl', 'SparseApplyFtrlV2', 'SparseApplyProximalAdagrad', 'SparseApplyRMSProp', 'SparseGatherV2', 'SparseSlice', 'SparseSoftmaxCrossEntropyWithLogits', 'SparseTensorDenseAdd', 'SparseTensorDenseMatmul', 'SparseToDense', 'Split', 'SplitV', 'Sqrt', 'Square', 'SquareSumAll', 'SquaredDifference', 'Squeeze', 'Stack', 'StandardLaplace', 'StandardNormal', 'StopGradient', 'StridedSlice', 'Sub', 'SubAndFilter', 'Svd', 'TBERegOp', 'Tan', 'Tanh', 'Tensor', 'TensorAdd', 'TensorScatterAdd', 'TensorScatterDiv', 'TensorScatterElements', 'TensorScatterMax', 'TensorScatterMin', 'TensorScatterMul', 'TensorScatterSub', 'TensorScatterUpdate', 'TensorShape', 'TensorSummary', 'Tile', 'TopK', 'Trace', 'Transpose', 'TridiagonalMatMul', 'Tril', 'TrilIndices', 'TripletMarginLoss', 'Triu', 'TriuIndices', 'Trunc', 'TruncateDiv', 'TruncateMod', 'TruncatedNormal', 'TupleToArray', 'UniformCandidateSampler', 'UniformInt', 'UniformReal', 'Unique', 'UniqueConsecutive', 'UniqueWithPad', 'Unpack', 'UnravelIndex', 'UnsortedSegmentMax', 'UnsortedSegmentMin', 'UnsortedSegmentProd', 'UnsortedSegmentSum', 'Unstack', 'UpdateState', 'UpperBound', 'UpsampleNearest3D', 'UpsampleTrilinear3D', 'Xdivy', 'Xlogy', 'Zeros', 'ZerosLike', 'Zeta', '_AllSwap', '_Grad', '_Vmap', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__primitive__', '__spec__', '_constants', '_op_impl', '_primitive_cache', '_register_for_op', '_utils', 'abs', 'absolute', 'absolute_import', 'accumulate_n', 'acos', 'acosh', 'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d', 'add', 'add_flags', 'addbmm', 'addcdiv', 'addcmul', 'addmm', 'addmv', 'addn', 'addr', 'adjoint', 'affine_grid', 'all', 'amax', 'amin', 'aminmax', 'angle', 'any', 'approximate_equal', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argsort', 'argwhere', 'array_ops', 'array_reduce', 'array_to_scalar', 'asin', 'asinh', 'assign', 'assign_add', 'assign_sub', 'atan', 'atan2', 'atanh', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'avg_pool1d', 'avg_pool2d', 'avg_pool3d', 'baddbmm', 'bartlett_window', 'batch_dot', 'batch_norm', 'batch_to_space_nd', 'bernoulli', 'bessel_i0', 'bessel_i0e', 'bessel_i1', 'bessel_i1e', 'bessel_j0', 'bessel_j1', 'bessel_k0', 'bessel_k0e', 'bessel_k1', 'bessel_k1e', 'bessel_y0', 'bessel_y1', 'bias_add', 'binary_cross_entropy', 'binary_cross_entropy_with_logits', 'bincount', 'bitwise_and', 'bitwise_left_shift', 'bitwise_or', 'bitwise_right_shift', 'bitwise_xor', 'blackman_window', 'block_diag', 'bmm', 'bool_and', 'bool_eq', 'bool_not', 'bool_or', 'bounding_box_decode', 'bounding_box_encode', 'broadcast_gradient_args', 'broadcast_to', 'cartesian_prod', 'cast', 'cat', 'cdist', 'ceil', 'celu', 'channel_shuffle', 'check_valid', 'choice_with_mask', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk', 'clamp', 'clip', 'clip_by_global_norm', 'clip_by_value', 'coalesce', 'col2im', 'column_stack', 'combinations', 'composite', 'concat', 'conj', 'constexpr', 'conv1d', 'conv2d', 'conv3d', 'conv3d_transpose', 'coo2csr', 'coo_abs', 'coo_acos', 'coo_acosh', 'coo_add', 'coo_asin', 'coo_asinh', 'coo_atan', 'coo_atanh', 'coo_ceil', 'coo_concat', 'coo_cos', 'coo_cosh', 'coo_exp', 'coo_expm1', 'coo_floor', 'coo_inv', 'coo_isfinite', 'coo_isinf', 'coo_isnan', 'coo_log', 'coo_log1p', 'coo_neg', 'coo_relu', 'coo_relu6', 'coo_round', 'coo_sigmoid', 'coo_sin', 'coo_sinh', 'coo_softsign', 'coo_sqrt', 'coo_square', 'coo_tan', 'coo_tanh', 'coo_tensor_get_dense_shape', 'coo_tensor_get_indices', 'coo_tensor_get_values', 'copysign', 'cos', 'cosh', 'cosine_embedding_loss', 'cosine_similarity', 'count_nonzero', 'cov', 'crop_and_resize', 'cross', 'cross_entropy', 'csr2coo', 'csr_abs', 'csr_acos', 'csr_acosh', 'csr_add', 'csr_asin', 'csr_asinh', 'csr_atan', 'csr_atanh', 'csr_ceil', 'csr_cos', 'csr_cosh', 'csr_div', 'csr_exp', 'csr_expm1', 'csr_floor', 'csr_gather', 'csr_inv', 'csr_isfinite', 'csr_isinf', 'csr_isnan', 'csr_log', 'csr_log1p', 'csr_mm', 'csr_mul', 'csr_mv', 'csr_neg', 'csr_reduce_sum', 'csr_relu', 'csr_relu6', 'csr_round', 'csr_sigmoid', 'csr_sin', 'csr_sinh', 'csr_softmax', 'csr_softsign', 'csr_sqrt', 'csr_square', 'csr_tan', 'csr_tanh', 'csr_tensor_get_dense_shape', 'csr_tensor_get_indices', 'csr_tensor_get_indptr', 'csr_tensor_get_values', 'csr_to_coo', 'csr_to_dense', 'ctc_greedy_decoder', 'ctc_loss', 'cummax', 'cummin', 'cumprod', 'cumsum', 'custom_info_register', 'deepcopy', 'deformable_conv2d', 'deg2rad', 'dense', 'dense_to_sparse_coo', 'dense_to_sparse_csr', 'depend', 'derivative', 'det', 'diag', 'diag_embed', 'diagflat', 'diagonal', 'dict_getitem', 'dict_setitem', 'diff', 'digamma', 'dist', 'distribute', 'div', 'divide', 'dot', 'dropout', 'dropout1d', 'dropout2d', 'dropout3d', 'dsplit', 'dstack', 'dtype', 'dyn_shape', 'eig', 'einsum', 'elu', 'embed', 'env_get', 'environ_add', 'environ_create', 'environ_get', 'environ_set', 'equal', 'erf', 'erfc', 'erfinv', 'exp', 'exp2', 'expand', 'expand_dims', 'expm1', 'eye', 'fast_gelu', 'fft', 'fft2', 'fftn', 'fill', 'fill_', 'fills', 'flatten', 'flip', 'fliplr', 'flipud', 'float_power', 'floor', 'floor_div', 'floor_mod', 'floordiv', 'floormod', 'fmax', 'fmin', 'fmod', 'fold', 'frac', 'fractional_max_pool2d', 'fractional_max_pool3d', 'full', 'full_like', 'function', 'functional', 'gamma', 'gather', 'gather_d', 'gather_elements', 'gather_nd', 'gaussian_nll_loss', 'gcd', 'ge', 'gelu', 'geqrf', 'ger', 'geswitch', 'get_grad', 'get_vm_impl_fn', 'glu', 'grad', 'greater', 'greater_equal', 'grid_sample', 'gt', 'gumbel_softmax', 'hamming_window', 'hann_window', 'hardshrink', 'hardsigmoid', 'hardswish', 'hardtanh', 'hastype', 'heaviside', 'hinge_embedding_loss', 'histc', 'hsplit', 'hstack', 'huber_loss', 'hyper_add', 'hypot', 'i0', 'identity', 'ifft', 'ifft2', 'ifftn', 'igamma', 'igammac', 'imag', 'in_dict', 'index_add', 'index_fill', 'index_select', 'inner', 'inplace_add', 'inplace_index_add', 'inplace_sub', 'inplace_update', 'interpolate', 'intopk', 'inv', 'inverse', 'invert', 'iou', 'is_', 'is_complex', 'is_dynamic_sequence_element_unknown', 'is_floating_point', 'is_not', 'is_sequence_shape_unknown', 'is_sequence_value_unknown', 'is_tensor', 'is_tensor_bool_cond', 'isclose', 'isconstant', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal', 'jacfwd', 'jacrev', 'jet', 'jvp', 'kaiser_window', 'kernel', 'kl_div', 'kron', 'l1_loss', 'laplace', 'lcm', 'ldexp', 'le', 'leaky_relu', 'lerp', 'less', 'less_equal', 'lgamma', 'linalg_ops', 'linearize', 'linspace', 'list_equal', 'list_getitem', 'list_len', 'list_setitem', 'log', 'log10', 'log1p', 'log2', 'log_matrix_determinant', 'log_softmax', 'log_uniform_candidate_sampler', 'logaddexp', 'logaddexp2', 'logdet', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logit', 'logsigmoid', 'logspace', 'logsumexp', 'lp_pool1d', 'lp_pool2d', 'lrn', 'lstsq', 'lt', 'lu_solve', 'lu_unpack', 'make_coo_tensor', 'make_csr_tensor', 'make_dict', 'make_list', 'make_map_parameter', 'make_range', 'make_row_tensor', 'make_row_tensor_inner', 'make_slice', 'make_sparse_tensor', 'make_tuple', 'margin_ranking_loss', 'masked_fill', 'masked_select', 'matmul', 'matrix_band_part', 'matrix_determinant', 'matrix_diag', 'matrix_diag_part', 'matrix_exp', 'matrix_power', 'matrix_set_diag', 'matrix_solve', 'max', 'max_pool2d', 'max_pool3d', 'max_unpool1d', 'max_unpool2d', 'max_unpool3d', 'maximum', 'mean', 'median', 'merge', 'meshgrid', 'min', 'minimum', 'mirror_pad', 'mish', 'mixed_precision_cast', 'mm', 'moveaxis', 'movedim', 'ms_kernel', 'mse_loss', 'msort', 'mul', 'multi_margin_loss', 'multilabel_margin_loss', 'multilabel_soft_margin_loss', 'multinomial', 'multinomial_with_replacement', 'multiply', 'mv', 'mvlgamma', 'nan_to_num', 'nanquantile', 'nansum', 'narrow', 'ne', 'neg', 'neg_tensor', 'negative', 'nextafter', 'nll_loss', 'nonzero', 'norm', 'normal', 'not_equal', 'not_in_dict', 'numel', 'one_hot', 'ones', 'ones_like', 'op_info_register', 'operations', 'orgqr', 'ormqr', 'outer', 'pad', 'padding', 'partial', 'pdist', 'permute', 'pinv', 'pixel_shuffle', 'pixel_unshuffle', 'poisson', 'polar', 'polygamma', 'population_count', 'positive', 'pow', 'pows', 'prelu', 'prim_attr_register', 'primitive', 'print_', 'prod', 'qr', 'quantile', 'rad2deg', 'rand', 'rand_like', 'randint', 'randint_like', 'randn', 'randn_like', 'random_categorical', 'random_gamma', 'random_poisson', 'randperm', 'range', 'rank', 'ravel', 'real', 'reciprocal', 'reduce_max', 'reduce_mean', 'reduce_min', 'reduce_sum', 'reduced_shape', 'ref_to_embed', 'relu', 'relu6', 'remainder', 'renorm', 'repeat_elements', 'repeat_interleave', 'reshape', 'reshape_', 'reverse', 'reverse_sequence', 'roll', 'rot90', 'round', 'row_tensor_add', 'row_tensor_get_dense_shape', 'row_tensor_get_indices', 'row_tensor_get_values', 'rrelu', 'rsqrt', 'scalar_add', 'scalar_cast', 'scalar_div', 'scalar_eq', 'scalar_floordiv', 'scalar_ge', 'scalar_gt', 'scalar_le', 'scalar_log', 'scalar_lt', 'scalar_mod', 'scalar_mul', 'scalar_ne', 'scalar_pow', 'scalar_sub', 'scalar_to_array', 'scalar_to_tensor', 'scalar_uadd', 'scalar_usub', 'scatter', 'scatter_add', 'scatter_div', 'scatter_max', 'scatter_min', 'scatter_mul', 'scatter_nd', 'scatter_nd_add', 'scatter_nd_div', 'scatter_nd_max', 'scatter_nd_min', 'scatter_nd_mul', 'scatter_nd_sub', 'scatter_nd_update', 'scatter_update', 'searchsorted', 'select', 'selu', 'sequence_mask', 'sgn', 'shape', 'shape_', 'shape_mul', 'shuffle', 'sigmoid', 'sign', 'signature', 'signbit', 'silu', 'sin', 'sinc', 'sinh', 'size', 'slice', 'slogdet', 'smooth_l1_loss', 'soft_shrink', 'softmax', 'softmin', 'softshrink', 'softsign', 'sort', 'space_to_batch_nd', 'sparse_segment_mean', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'standard_laplace', 'standard_normal', 'std', 'std_mean', 'stft', 'stop_gradient', 'strided_slice', 'string_concat', 'string_eq', 'sub', 'subtract', 'sum', 'svd', 'swapaxes', 'swapdims', 'switch', 'switch_layer', 't', 'tail', 'tan', 'tanh', 'tanhshrink', 'tensor_add', 'tensor_div', 'tensor_dot', 'tensor_exp', 'tensor_expm1', 'tensor_floordiv', 'tensor_ge', 'tensor_gt', 'tensor_le', 'tensor_lt', 'tensor_mod', 'tensor_mul', 'tensor_operator_registry', 'tensor_pow', 'tensor_range', 'tensor_scatter_add', 'tensor_scatter_div', 'tensor_scatter_elements', 'tensor_scatter_max', 'tensor_scatter_min', 'tensor_scatter_mul', 'tensor_scatter_sub', 'tensor_scatter_update', 'tensor_slice', 'tensor_split', 'tensor_sub', 'threshold', 'tile', 'top_k', 'topk', 'trace', 'transpose', 'trapz', 'tril', 'tril_indices', 'triplet_margin_loss', 'triu', 'triu_indices', 'true_divide', 'trunc', 'truncate_div', 'truncate_mod', 'tuple_div', 'tuple_equal', 'tuple_getitem', 'tuple_len', 'tuple_reversed', 'tuple_setitem', 'tuple_to_array', 'typeof', 'unbind', 'unfold', 'uniform', 'uniform_candidate_sampler', 'unique', 'unique_consecutive', 'unique_with_pad', 'unsorted_segment_max', 'unsorted_segment_min', 'unsorted_segment_prod', 'unsorted_segment_sum', 'unsqueeze', 'unstack', 'upsample', 'value_and_grad', 'var', 'var_mean', 'view_as_real', 'vjp', 'vm_impl_registry', 'vmap', 'vsplit', 'vstack', 'where', 'xdivy', 'xlogy', 'zeros', 'zeros_like', 'zeta', 'zip_operation']


通常可以忽略以“`__`”（双下划线）开始和结束的函数，它们是Python中的特殊对象，
或以单个“`_`”（单下划线）开始的函数，它们通常是内部函数。
根据剩余的函数名或属性名，我们可能会猜测这个模块提供了各种生成随机数的方法，
包括从均匀分布（`uniform`）、正态分布（`normal`）和多项分布（`multinomial`）中采样。

## 查找特定函数和类的用法

有关如何使用给定函数或类的更具体说明，可以调用`help`函数。
例如，我们来[**查看张量`ones`函数的用法。**]



```python
from mindspore import ops
```


```python
help(ops.ones)
```

    Help on function ones in module mindspore.ops.function.array_func:
    
    ones(shape, dtype=None)
        Creates a tensor filled with value ones.
        
        Creates a tensor with shape described by the first argument and fills it with value ones in type of the second
        argument.
        
        Args:
            shape (Union[tuple[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
                tuple or Tensor containing positive integers are allowed. If it is a Tensor,
                it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.
            dtype (:class:`mindspore.dtype`): The specified type of output tensor. If `dtype` is None,
                `mindspore.float32` will be used. Default: None.
        
        Returns:
            Tensor, has the same type and shape as input shape value.
        
        Raises:
            TypeError: If `shape` is not tuple, int or Tensor.
        
        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
        
        Examples:
            >>> output = ops.ones((2, 2), mindspore.float32)
            >>> print(output)
            [[1. 1.]
             [1. 1.]]
    


从文档中，我们可以看到`ones`函数创建一个具有指定形状的新张量，并将所有元素值设置为1。
下面来[**运行一个快速测试**]来确认这一解释：



```python
ops.ones(4)
```




    Tensor(shape=[4], dtype=Float32, value= [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00])



在Jupyter记事本中，我们可以使用`?`指令在另一个浏览器窗口中显示文档。
例如，`list?`指令将创建与`help(list)`指令几乎相同的内容，并在新的浏览器窗口中显示它。
此外，如果我们使用两个问号，如`list??`，将显示实现该函数的Python代码。

## 小结

* 官方文档提供了本书之外的大量描述和示例。
* 可以通过调用`dir`和`help`函数或在Jupyter记事本中使用`?`和`??`查看API的用法文档。

## 练习

1. 在深度学习框架中查找任何函数或类的文档。请尝试在这个框架的官方网站上找到文档。


[Discussions](https://discuss.d2l.ai/t/1765)

