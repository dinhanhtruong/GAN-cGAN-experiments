??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
#cond_generator/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2*4
shared_name%#cond_generator/embedding/embeddings
?
7cond_generator/embedding/embeddings/Read/ReadVariableOpReadVariableOp#cond_generator/embedding/embeddings*
_output_shapes

:
2*
dtype0
?
cond_generator/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:21*,
shared_namecond_generator/dense/kernel
?
/cond_generator/dense/kernel/Read/ReadVariableOpReadVariableOpcond_generator/dense/kernel*
_output_shapes

:21*
dtype0
?
cond_generator/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1**
shared_namecond_generator/dense/bias
?
-cond_generator/dense/bias/Read/ReadVariableOpReadVariableOpcond_generator/dense/bias*
_output_shapes
:1*
dtype0
?
module_wrapper/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?1*.
shared_namemodule_wrapper/dense_1/kernel
?
1module_wrapper/dense_1/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper/dense_1/kernel*
_output_shapes
:	d?1*
dtype0
?
module_wrapper/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?1*,
shared_namemodule_wrapper/dense_1/bias
?
/module_wrapper/dense_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper/dense_1/bias*
_output_shapes	
:?1*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameconv2d_transpose/bias
|
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv2d_transpose_1/bias
?
+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes	
:?*
dtype0
?
module_wrapper_5/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name module_wrapper_5/conv2d/kernel
?
2module_wrapper_5/conv2d/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_5/conv2d/kernel*'
_output_shapes
:?*
dtype0
?
module_wrapper_5/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namemodule_wrapper_5/conv2d/bias
?
0module_wrapper_5/conv2d/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_5/conv2d/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?9
value?9B?9 B?9
?
emb
	emb_dense
transform_latent_vecs
	model
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
b


embeddings
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
layer_with_weights-0
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
 layer_with_weights-2
 layer-4
!	variables
"trainable_variables
#regularization_losses
$	keras_api
N

0
1
2
%3
&4
'5
(6
)7
*8
+9
,10
N

0
1
2
%3
&4
'5
(6
)7
*8
+9
,10
 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
 
b`
VARIABLE_VALUE#cond_generator/embedding/embeddings)emb/embeddings/.ATTRIBUTES/VARIABLE_VALUE


0


0
 
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEcond_generator/dense/kernel+emb_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEcond_generator/dense/bias)emb_dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
_
<_module
=	variables
>trainable_variables
?regularization_losses
@	keras_api
_
A_module
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
_
F_module
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api

%0
&1

%0
&1
 
?

Klayers
	variables
Llayer_metrics
trainable_variables
Mlayer_regularization_losses
regularization_losses
Nmetrics
Onon_trainable_variables
h

'kernel
(bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
_
T_module
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

)kernel
*bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
_
]_module
^	variables
_trainable_variables
`regularization_losses
a	keras_api
_
b_module
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
*
'0
(1
)2
*3
+4
,5
*
'0
(1
)2
*3
+4
,5
 
?

glayers
!	variables
hlayer_metrics
"trainable_variables
ilayer_regularization_losses
#regularization_losses
jmetrics
knon_trainable_variables
YW
VARIABLE_VALUEmodule_wrapper/dense_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEmodule_wrapper/dense_1/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_transpose/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d_transpose/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_transpose_1/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_transpose_1/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEmodule_wrapper_5/conv2d/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmodule_wrapper_5/conv2d/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
h

%kernel
&bias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api

%0
&1

%0
&1
 
?

players
=	variables
qlayer_metrics
>trainable_variables
rlayer_regularization_losses
?regularization_losses
smetrics
tnon_trainable_variables
R
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
 
 
 
?

ylayers
B	variables
zlayer_metrics
Ctrainable_variables
{layer_regularization_losses
Dregularization_losses
|metrics
}non_trainable_variables
T
~	variables
trainable_variables
?regularization_losses
?	keras_api
 
 
 
?
?layers
G	variables
?layer_metrics
Htrainable_variables
 ?layer_regularization_losses
Iregularization_losses
?metrics
?non_trainable_variables

0
1
2
 
 
 
 

'0
(1

'0
(1
 
?
?layers
P	variables
?layer_metrics
Qtrainable_variables
 ?layer_regularization_losses
Rregularization_losses
?metrics
?non_trainable_variables
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
 
 
?
?layers
U	variables
?layer_metrics
Vtrainable_variables
 ?layer_regularization_losses
Wregularization_losses
?metrics
?non_trainable_variables

)0
*1

)0
*1
 
?
?layers
Y	variables
?layer_metrics
Ztrainable_variables
 ?layer_regularization_losses
[regularization_losses
?metrics
?non_trainable_variables
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
 
 
?
?layers
^	variables
?layer_metrics
_trainable_variables
 ?layer_regularization_losses
`regularization_losses
?metrics
?non_trainable_variables
l

+kernel
,bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

+0
,1

+0
,1
 
?
?layers
c	variables
?layer_metrics
dtrainable_variables
 ?layer_regularization_losses
eregularization_losses
?metrics
?non_trainable_variables
#
0
1
2
3
 4
 
 
 
 

%0
&1

%0
&1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 
 
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
 
 
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
 
 

+0
,1

+0
,1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
y
serving_default_args_0Placeholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
y
serving_default_args_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_1#cond_generator/embedding/embeddingscond_generator/dense/kernelcond_generator/dense/biasmodule_wrapper/dense_1/kernelmodule_wrapper/dense_1/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasmodule_wrapper_5/conv2d/kernelmodule_wrapper_5/conv2d/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_18847
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7cond_generator/embedding/embeddings/Read/ReadVariableOp/cond_generator/dense/kernel/Read/ReadVariableOp-cond_generator/dense/bias/Read/ReadVariableOp1module_wrapper/dense_1/kernel/Read/ReadVariableOp/module_wrapper/dense_1/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp2module_wrapper_5/conv2d/kernel/Read/ReadVariableOp0module_wrapper_5/conv2d/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_19977
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#cond_generator/embedding/embeddingscond_generator/dense/kernelcond_generator/dense/biasmodule_wrapper/dense_1/kernelmodule_wrapper/dense_1/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasmodule_wrapper_5/conv2d/kernelmodule_wrapper_5/conv2d/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_20020??
?
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19845

args_0
identity^
leaky_re_lu_1/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
L
0__inference_module_wrapper_1_layer_call_fn_19797

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18875a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?0
?
!__inference__traced_restore_20020
file_prefixF
4assignvariableop_cond_generator_embedding_embeddings:
2@
.assignvariableop_1_cond_generator_dense_kernel:21:
,assignvariableop_2_cond_generator_dense_bias:1C
0assignvariableop_3_module_wrapper_dense_1_kernel:	d?1=
.assignvariableop_4_module_wrapper_dense_1_bias:	?1F
*assignvariableop_5_conv2d_transpose_kernel:??7
(assignvariableop_6_conv2d_transpose_bias:	?H
,assignvariableop_7_conv2d_transpose_1_kernel:??9
*assignvariableop_8_conv2d_transpose_1_bias:	?L
1assignvariableop_9_module_wrapper_5_conv2d_kernel:?>
0assignvariableop_10_module_wrapper_5_conv2d_bias:
identity_12??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)emb/embeddings/.ATTRIBUTES/VARIABLE_VALUEB+emb_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)emb_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp4assignvariableop_cond_generator_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp.assignvariableop_1_cond_generator_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_cond_generator_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp0assignvariableop_3_module_wrapper_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_module_wrapper_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_conv2d_transpose_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_conv2d_transpose_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv2d_transpose_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp1assignvariableop_9_module_wrapper_5_conv2d_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_module_wrapper_5_conv2d_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_12Identity_12:output:0*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
0__inference_conv2d_transpose_layer_call_fn_19075

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_19548

inputs3
!tensordot_readvariableop_resource:21-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:21*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:1Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19242

args_0
identity^
leaky_re_lu_1/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_19031
module_wrapper_input'
module_wrapper_19023:	d?1#
module_wrapper_19025:	?1
identity??&module_wrapper/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_19023module_wrapper_19025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18964?
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18939?
 module_wrapper_2/PartitionedCallPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18923?
IdentityIdentity)module_wrapper_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????o
NoOpNoOp'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:] Y
'
_output_shapes
:?????????d
.
_user_specified_namemodule_wrapper_input
?
?
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19902

args_0@
%conv2d_conv2d_readvariableop_resource:?4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
IdentityIdentityconv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
*__inference_sequential_layer_call_fn_19599

inputs
unknown:	d?1
	unknown_0:	?1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18894x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_18699

inputs3
!tensordot_readvariableop_resource:21-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:21*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:1Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
g
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19148

args_0
identity^
leaky_re_lu_2/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?	
?
,__inference_sequential_1_layer_call_fn_19744

inputs#
unknown:??
	unknown_0:	?%
	unknown_1:??
	unknown_2:	?$
	unknown_3:?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19285w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
L
0__inference_module_wrapper_1_layer_call_fn_19802

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18939a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?

?
I__inference_module_wrapper_layer_call_and_return_conditional_losses_19754

args_09
&dense_1_matmul_readvariableop_resource:	d?16
'dense_1_biasadd_readvariableop_resource:	?1
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0z
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1h
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
L
0__inference_module_wrapper_2_layer_call_fn_19835

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18891i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?	
?
D__inference_embedding_layer_call_and_return_conditional_losses_19509

inputs(
embedding_lookup_19503:
2
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookupResourceGatherembedding_lookup_19503Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/19503*+
_output_shapes
:?????????2*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/19503*+
_output_shapes
:?????????2?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19359
conv2d_transpose_input2
conv2d_transpose_19341:??%
conv2d_transpose_19343:	?4
conv2d_transpose_1_19347:??'
conv2d_transpose_1_19349:	?1
module_wrapper_5_19353:?$
module_wrapper_5_19355:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputconv2d_transpose_19341conv2d_transpose_19343*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065?
 module_wrapper_3/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19242?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_19347conv2d_transpose_1_19349*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109?
 module_wrapper_4/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19226?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_19353module_wrapper_5_19355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19206?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:h d
0
_output_shapes
:??????????
0
_user_specified_nameconv2d_transpose_input
?
L
0__inference_module_wrapper_4_layer_call_fn_19875

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19148i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
? 
?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
I__inference_module_wrapper_layer_call_and_return_conditional_losses_19764

args_09
&dense_1_matmul_readvariableop_resource:	d?16
'dense_1_biasadd_readvariableop_resource:	?1
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0z
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1h
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18875

args_0
identityT
leaky_re_lu/LeakyRelu	LeakyReluargs_0*(
_output_shapes
:??????????1l
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?	
?
,__inference_sequential_1_layer_call_fn_19317
conv2d_transpose_input#
unknown:??
	unknown_0:	?%
	unknown_1:??
	unknown_2:	?$
	unknown_3:?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19285w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:??????????
0
_user_specified_nameconv2d_transpose_input
??
?
 __inference__wrapped_model_18646

args_0

args_1A
/cond_generator_embedding_embedding_lookup_18466:
2H
6cond_generator_dense_tensordot_readvariableop_resource:21B
4cond_generator_dense_biasadd_readvariableop_resource:1b
Ocond_generator_sequential_module_wrapper_dense_1_matmul_readvariableop_resource:	d?1_
Pcond_generator_sequential_module_wrapper_dense_1_biasadd_readvariableop_resource:	?1q
Ucond_generator_sequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:??[
Lcond_generator_sequential_1_conv2d_transpose_biasadd_readvariableop_resource:	?s
Wcond_generator_sequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??]
Ncond_generator_sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource:	?m
Rcond_generator_sequential_1_module_wrapper_5_conv2d_conv2d_readvariableop_resource:?a
Scond_generator_sequential_1_module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??+cond_generator/dense/BiasAdd/ReadVariableOp?-cond_generator/dense/Tensordot/ReadVariableOp?)cond_generator/embedding/embedding_lookup?Gcond_generator/sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp?Fcond_generator/sequential/module_wrapper/dense_1/MatMul/ReadVariableOp?Ccond_generator/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?Lcond_generator/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?Econd_generator/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp?Ncond_generator/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?Jcond_generator/sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?Icond_generator/sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOpn
cond_generator/embedding/CastCastargs_1*

DstT0*

SrcT0*'
_output_shapes
:??????????
)cond_generator/embedding/embedding_lookupResourceGather/cond_generator_embedding_embedding_lookup_18466!cond_generator/embedding/Cast:y:0*
Tindices0*B
_class8
64loc:@cond_generator/embedding/embedding_lookup/18466*+
_output_shapes
:?????????2*
dtype0?
2cond_generator/embedding/embedding_lookup/IdentityIdentity2cond_generator/embedding/embedding_lookup:output:0*
T0*B
_class8
64loc:@cond_generator/embedding/embedding_lookup/18466*+
_output_shapes
:?????????2?
4cond_generator/embedding/embedding_lookup/Identity_1Identity;cond_generator/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2?
-cond_generator/dense/Tensordot/ReadVariableOpReadVariableOp6cond_generator_dense_tensordot_readvariableop_resource*
_output_shapes

:21*
dtype0m
#cond_generator/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#cond_generator/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
$cond_generator/dense/Tensordot/ShapeShape=cond_generator/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:n
,cond_generator/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'cond_generator/dense/Tensordot/GatherV2GatherV2-cond_generator/dense/Tensordot/Shape:output:0,cond_generator/dense/Tensordot/free:output:05cond_generator/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.cond_generator/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
)cond_generator/dense/Tensordot/GatherV2_1GatherV2-cond_generator/dense/Tensordot/Shape:output:0,cond_generator/dense/Tensordot/axes:output:07cond_generator/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$cond_generator/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
#cond_generator/dense/Tensordot/ProdProd0cond_generator/dense/Tensordot/GatherV2:output:0-cond_generator/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&cond_generator/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
%cond_generator/dense/Tensordot/Prod_1Prod2cond_generator/dense/Tensordot/GatherV2_1:output:0/cond_generator/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*cond_generator/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%cond_generator/dense/Tensordot/concatConcatV2,cond_generator/dense/Tensordot/free:output:0,cond_generator/dense/Tensordot/axes:output:03cond_generator/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$cond_generator/dense/Tensordot/stackPack,cond_generator/dense/Tensordot/Prod:output:0.cond_generator/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
(cond_generator/dense/Tensordot/transpose	Transpose=cond_generator/embedding/embedding_lookup/Identity_1:output:0.cond_generator/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2?
&cond_generator/dense/Tensordot/ReshapeReshape,cond_generator/dense/Tensordot/transpose:y:0-cond_generator/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
%cond_generator/dense/Tensordot/MatMulMatMul/cond_generator/dense/Tensordot/Reshape:output:05cond_generator/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1p
&cond_generator/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:1n
,cond_generator/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
'cond_generator/dense/Tensordot/concat_1ConcatV20cond_generator/dense/Tensordot/GatherV2:output:0/cond_generator/dense/Tensordot/Const_2:output:05cond_generator/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
cond_generator/dense/TensordotReshape/cond_generator/dense/Tensordot/MatMul:product:00cond_generator/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1?
+cond_generator/dense/BiasAdd/ReadVariableOpReadVariableOp4cond_generator_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0?
cond_generator/dense/BiasAddBiasAdd'cond_generator/dense/Tensordot:output:03cond_generator/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1u
cond_generator/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
cond_generator/ReshapeReshape%cond_generator/dense/BiasAdd:output:0%cond_generator/Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
Fcond_generator/sequential/module_wrapper/dense_1/MatMul/ReadVariableOpReadVariableOpOcond_generator_sequential_module_wrapper_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0?
7cond_generator/sequential/module_wrapper/dense_1/MatMulMatMulargs_0Ncond_generator/sequential/module_wrapper/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
Gcond_generator/sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOpReadVariableOpPcond_generator_sequential_module_wrapper_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
8cond_generator/sequential/module_wrapper/dense_1/BiasAddBiasAddAcond_generator/sequential/module_wrapper/dense_1/MatMul:product:0Ocond_generator/sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
@cond_generator/sequential/module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyReluAcond_generator/sequential/module_wrapper/dense_1/BiasAdd:output:0*(
_output_shapes
:??????????1?
8cond_generator/sequential/module_wrapper_2/reshape/ShapeShapeNcond_generator/sequential/module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:?
Fcond_generator/sequential/module_wrapper_2/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Hcond_generator/sequential/module_wrapper_2/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hcond_generator/sequential/module_wrapper_2/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@cond_generator/sequential/module_wrapper_2/reshape/strided_sliceStridedSliceAcond_generator/sequential/module_wrapper_2/reshape/Shape:output:0Ocond_generator/sequential/module_wrapper_2/reshape/strided_slice/stack:output:0Qcond_generator/sequential/module_wrapper_2/reshape/strided_slice/stack_1:output:0Qcond_generator/sequential/module_wrapper_2/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Bcond_generator/sequential/module_wrapper_2/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
Bcond_generator/sequential/module_wrapper_2/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Bcond_generator/sequential/module_wrapper_2/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
@cond_generator/sequential/module_wrapper_2/reshape/Reshape/shapePackIcond_generator/sequential/module_wrapper_2/reshape/strided_slice:output:0Kcond_generator/sequential/module_wrapper_2/reshape/Reshape/shape/1:output:0Kcond_generator/sequential/module_wrapper_2/reshape/Reshape/shape/2:output:0Kcond_generator/sequential/module_wrapper_2/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
:cond_generator/sequential/module_wrapper_2/reshape/ReshapeReshapeNcond_generator/sequential/module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0Icond_generator/sequential/module_wrapper_2/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????\
cond_generator/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
cond_generator/concatConcatV2Ccond_generator/sequential/module_wrapper_2/reshape/Reshape:output:0cond_generator/Reshape:output:0#cond_generator/concat/axis:output:0*
N*
T0*0
_output_shapes
:???????????
2cond_generator/sequential_1/conv2d_transpose/ShapeShapecond_generator/concat:output:0*
T0*
_output_shapes
:?
@cond_generator/sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bcond_generator/sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bcond_generator/sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:cond_generator/sequential_1/conv2d_transpose/strided_sliceStridedSlice;cond_generator/sequential_1/conv2d_transpose/Shape:output:0Icond_generator/sequential_1/conv2d_transpose/strided_slice/stack:output:0Kcond_generator/sequential_1/conv2d_transpose/strided_slice/stack_1:output:0Kcond_generator/sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4cond_generator/sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :v
4cond_generator/sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :w
4cond_generator/sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
2cond_generator/sequential_1/conv2d_transpose/stackPackCcond_generator/sequential_1/conv2d_transpose/strided_slice:output:0=cond_generator/sequential_1/conv2d_transpose/stack/1:output:0=cond_generator/sequential_1/conv2d_transpose/stack/2:output:0=cond_generator/sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:?
Bcond_generator/sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dcond_generator/sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dcond_generator/sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<cond_generator/sequential_1/conv2d_transpose/strided_slice_1StridedSlice;cond_generator/sequential_1/conv2d_transpose/stack:output:0Kcond_generator/sequential_1/conv2d_transpose/strided_slice_1/stack:output:0Mcond_generator/sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0Mcond_generator/sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Lcond_generator/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpUcond_generator_sequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
=cond_generator/sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput;cond_generator/sequential_1/conv2d_transpose/stack:output:0Tcond_generator/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0cond_generator/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
Ccond_generator/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpLcond_generator_sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
4cond_generator/sequential_1/conv2d_transpose/BiasAddBiasAddFcond_generator/sequential_1/conv2d_transpose/conv2d_transpose:output:0Kcond_generator/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
Dcond_generator/sequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu	LeakyRelu=cond_generator/sequential_1/conv2d_transpose/BiasAdd:output:0*0
_output_shapes
:???????????
4cond_generator/sequential_1/conv2d_transpose_1/ShapeShapeRcond_generator/sequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:?
Bcond_generator/sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Dcond_generator/sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Dcond_generator/sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
<cond_generator/sequential_1/conv2d_transpose_1/strided_sliceStridedSlice=cond_generator/sequential_1/conv2d_transpose_1/Shape:output:0Kcond_generator/sequential_1/conv2d_transpose_1/strided_slice/stack:output:0Mcond_generator/sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0Mcond_generator/sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6cond_generator/sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :x
6cond_generator/sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :y
6cond_generator/sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
4cond_generator/sequential_1/conv2d_transpose_1/stackPackEcond_generator/sequential_1/conv2d_transpose_1/strided_slice:output:0?cond_generator/sequential_1/conv2d_transpose_1/stack/1:output:0?cond_generator/sequential_1/conv2d_transpose_1/stack/2:output:0?cond_generator/sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:?
Dcond_generator/sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fcond_generator/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fcond_generator/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>cond_generator/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice=cond_generator/sequential_1/conv2d_transpose_1/stack:output:0Mcond_generator/sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0Ocond_generator/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ocond_generator/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ncond_generator/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpWcond_generator_sequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
?cond_generator/sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput=cond_generator/sequential_1/conv2d_transpose_1/stack:output:0Vcond_generator/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Rcond_generator/sequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
Econd_generator/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpNcond_generator_sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
6cond_generator/sequential_1/conv2d_transpose_1/BiasAddBiasAddHcond_generator/sequential_1/conv2d_transpose_1/conv2d_transpose:output:0Mcond_generator/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
Dcond_generator/sequential_1/module_wrapper_4/leaky_re_lu_2/LeakyRelu	LeakyRelu?cond_generator/sequential_1/conv2d_transpose_1/BiasAdd:output:0*0
_output_shapes
:???????????
Icond_generator/sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOpReadVariableOpRcond_generator_sequential_1_module_wrapper_5_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
:cond_generator/sequential_1/module_wrapper_5/conv2d/Conv2DConv2DRcond_generator/sequential_1/module_wrapper_4/leaky_re_lu_2/LeakyRelu:activations:0Qcond_generator/sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
Jcond_generator/sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOpReadVariableOpScond_generator_sequential_1_module_wrapper_5_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
;cond_generator/sequential_1/module_wrapper_5/conv2d/BiasAddBiasAddCcond_generator/sequential_1/module_wrapper_5/conv2d/Conv2D:output:0Rcond_generator/sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
8cond_generator/sequential_1/module_wrapper_5/conv2d/TanhTanhDcond_generator/sequential_1/module_wrapper_5/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
IdentityIdentity<cond_generator/sequential_1/module_wrapper_5/conv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^cond_generator/dense/BiasAdd/ReadVariableOp.^cond_generator/dense/Tensordot/ReadVariableOp*^cond_generator/embedding/embedding_lookupH^cond_generator/sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOpG^cond_generator/sequential/module_wrapper/dense_1/MatMul/ReadVariableOpD^cond_generator/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpM^cond_generator/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpF^cond_generator/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpO^cond_generator/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpK^cond_generator/sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOpJ^cond_generator/sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????d:?????????: : : : : : : : : : : 2Z
+cond_generator/dense/BiasAdd/ReadVariableOp+cond_generator/dense/BiasAdd/ReadVariableOp2^
-cond_generator/dense/Tensordot/ReadVariableOp-cond_generator/dense/Tensordot/ReadVariableOp2V
)cond_generator/embedding/embedding_lookup)cond_generator/embedding/embedding_lookup2?
Gcond_generator/sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOpGcond_generator/sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp2?
Fcond_generator/sequential/module_wrapper/dense_1/MatMul/ReadVariableOpFcond_generator/sequential/module_wrapper/dense_1/MatMul/ReadVariableOp2?
Ccond_generator/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpCcond_generator/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2?
Lcond_generator/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpLcond_generator/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2?
Econd_generator/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpEcond_generator/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Ncond_generator/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpNcond_generator/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2?
Jcond_generator/sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOpJcond_generator/sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2?
Icond_generator/sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOpIcond_generator/sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
?
?
.__inference_module_wrapper_layer_call_fn_19773

args_0
unknown:	d?1
	unknown_0:	?1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18864p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19168

inputs2
conv2d_transpose_19126:??%
conv2d_transpose_19128:	?4
conv2d_transpose_1_19138:??'
conv2d_transpose_1_19140:	?1
module_wrapper_5_19162:?$
module_wrapper_5_19164:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_19126conv2d_transpose_19128*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065?
 module_wrapper_3/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19136?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_19138conv2d_transpose_1_19140*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109?
 module_wrapper_4/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19148?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_19162module_wrapper_5_19164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19161?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19161

args_0@
%conv2d_conv2d_readvariableop_resource:?4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
IdentityIdentityconv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?	
?
,__inference_sequential_1_layer_call_fn_19183
conv2d_transpose_input#
unknown:??
	unknown_0:	?%
	unknown_1:??
	unknown_2:	?$
	unknown_3:?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19168w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:??????????
0
_user_specified_nameconv2d_transpose_input
?
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19136

args_0
identity^
leaky_re_lu_1/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
g
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_19816

args_0
identityC
reshape/ShapeShapeargs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:}
reshape/ReshapeReshapeargs_0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????i
IdentityIdentityreshape/Reshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_19569

inputsH
5module_wrapper_dense_1_matmul_readvariableop_resource:	d?1E
6module_wrapper_dense_1_biasadd_readvariableop_resource:	?1
identity??-module_wrapper/dense_1/BiasAdd/ReadVariableOp?,module_wrapper/dense_1/MatMul/ReadVariableOp?
,module_wrapper/dense_1/MatMul/ReadVariableOpReadVariableOp5module_wrapper_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0?
module_wrapper/dense_1/MatMulMatMulinputs4module_wrapper/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
-module_wrapper/dense_1/BiasAdd/ReadVariableOpReadVariableOp6module_wrapper_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
module_wrapper/dense_1/BiasAddBiasAdd'module_wrapper/dense_1/MatMul:product:05module_wrapper/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
&module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyRelu'module_wrapper/dense_1/BiasAdd:output:0*(
_output_shapes
:??????????1?
module_wrapper_2/reshape/ShapeShape4module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:v
,module_wrapper_2/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.module_wrapper_2/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.module_wrapper_2/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&module_wrapper_2/reshape/strided_sliceStridedSlice'module_wrapper_2/reshape/Shape:output:05module_wrapper_2/reshape/strided_slice/stack:output:07module_wrapper_2/reshape/strided_slice/stack_1:output:07module_wrapper_2/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(module_wrapper_2/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :j
(module_wrapper_2/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
(module_wrapper_2/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
&module_wrapper_2/reshape/Reshape/shapePack/module_wrapper_2/reshape/strided_slice:output:01module_wrapper_2/reshape/Reshape/shape/1:output:01module_wrapper_2/reshape/Reshape/shape/2:output:01module_wrapper_2/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
 module_wrapper_2/reshape/ReshapeReshape4module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0/module_wrapper_2/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:???????????
IdentityIdentity)module_wrapper_2/reshape/Reshape:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp.^module_wrapper/dense_1/BiasAdd/ReadVariableOp-^module_wrapper/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2^
-module_wrapper/dense_1/BiasAdd/ReadVariableOp-module_wrapper/dense_1/BiasAdd/ReadVariableOp2\
,module_wrapper/dense_1/MatMul/ReadVariableOp,module_wrapper/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?D
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19659

inputsU
9conv2d_transpose_conv2d_transpose_readvariableop_resource:???
0conv2d_transpose_biasadd_readvariableop_resource:	?W
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??A
2conv2d_transpose_1_biasadd_readvariableop_resource:	?Q
6module_wrapper_5_conv2d_conv2d_readvariableop_resource:?E
7module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?-module_wrapper_5/conv2d/Conv2D/ReadVariableOpL
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
(module_wrapper_3/leaky_re_lu_1/LeakyRelu	LeakyRelu!conv2d_transpose/BiasAdd:output:0*0
_output_shapes
:??????????~
conv2d_transpose_1/ShapeShape6module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:06module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
(module_wrapper_4/leaky_re_lu_2/LeakyRelu	LeakyRelu#conv2d_transpose_1/BiasAdd:output:0*0
_output_shapes
:???????????
-module_wrapper_5/conv2d/Conv2D/ReadVariableOpReadVariableOp6module_wrapper_5_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
module_wrapper_5/conv2d/Conv2DConv2D6module_wrapper_4/leaky_re_lu_2/LeakyRelu:activations:05module_wrapper_5/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.module_wrapper_5/conv2d/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_5_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
module_wrapper_5/conv2d/BiasAddBiasAdd'module_wrapper_5/conv2d/Conv2D:output:06module_wrapper_5/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
module_wrapper_5/conv2d/TanhTanh(module_wrapper_5/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????w
IdentityIdentity module_wrapper_5/conv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp/^module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.^module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2`
.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2^
-module_wrapper_5/conv2d/Conv2D/ReadVariableOp-module_wrapper_5/conv2d/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_module_wrapper_layer_call_fn_19782

args_0
unknown:	d?1
	unknown_0:	?1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18964p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?#
?
__inference__traced_save_19977
file_prefixB
>savev2_cond_generator_embedding_embeddings_read_readvariableop:
6savev2_cond_generator_dense_kernel_read_readvariableop8
4savev2_cond_generator_dense_bias_read_readvariableop<
8savev2_module_wrapper_dense_1_kernel_read_readvariableop:
6savev2_module_wrapper_dense_1_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop=
9savev2_module_wrapper_5_conv2d_kernel_read_readvariableop;
7savev2_module_wrapper_5_conv2d_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)emb/embeddings/.ATTRIBUTES/VARIABLE_VALUEB+emb_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)emb_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_cond_generator_embedding_embeddings_read_readvariableop6savev2_cond_generator_dense_kernel_read_readvariableop4savev2_cond_generator_dense_bias_read_readvariableop8savev2_module_wrapper_dense_1_kernel_read_readvariableop6savev2_module_wrapper_dense_1_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop9savev2_module_wrapper_5_conv2d_kernel_read_readvariableop7savev2_module_wrapper_5_conv2d_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
: :
2:21:1:	d?1:?1:??:?:??:?:?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
2:$ 

_output_shapes

:21: 

_output_shapes
:1:%!

_output_shapes
:	d?1:!

_output_shapes	
:?1:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!	

_output_shapes	
:?:-
)
'
_output_shapes
:?: 

_output_shapes
::

_output_shapes
: 
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19338
conv2d_transpose_input2
conv2d_transpose_19320:??%
conv2d_transpose_19322:	?4
conv2d_transpose_1_19326:??'
conv2d_transpose_1_19328:	?1
module_wrapper_5_19332:?$
module_wrapper_5_19334:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallconv2d_transpose_inputconv2d_transpose_19320conv2d_transpose_19322*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065?
 module_wrapper_3/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19136?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_19326conv2d_transpose_1_19328*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109?
 module_wrapper_4/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19148?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_19332module_wrapper_5_19334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19161?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:h d
0
_output_shapes
:??????????
0
_user_specified_nameconv2d_transpose_input
?
g
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18891

args_0
identityC
reshape/ShapeShapeargs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:}
reshape/ReshapeReshapeargs_0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????i
IdentityIdentityreshape/Reshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?
L
0__inference_module_wrapper_4_layer_call_fn_19880

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19226i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
g
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19865

args_0
identity^
leaky_re_lu_2/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
%__inference_dense_layer_call_fn_19518

inputs
unknown:21
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18699s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????2: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

?
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18964

args_09
&dense_1_matmul_readvariableop_resource:	d?16
'dense_1_biasadd_readvariableop_resource:	?1
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0z
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1h
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
?
#__inference_signature_wrapper_18847

args_0

args_1
unknown:
2
	unknown_0:21
	unknown_1:1
	unknown_2:	d?1
	unknown_3:	?1%
	unknown_4:??
	unknown_5:	?%
	unknown_6:??
	unknown_7:	?$
	unknown_8:?
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0args_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_18646w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????d:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0:OK
'
_output_shapes
:?????????
 
_user_specified_nameargs_1
?
g
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19226

args_0
identity^
leaky_re_lu_2/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19285

inputs2
conv2d_transpose_19267:??%
conv2d_transpose_19269:	?4
conv2d_transpose_1_19273:??'
conv2d_transpose_1_19275:	?1
module_wrapper_5_19279:?$
module_wrapper_5_19281:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_19267conv2d_transpose_19269*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065?
 module_wrapper_3/PartitionedCallPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19242?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_19273conv2d_transpose_1_19275*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109?
 module_wrapper_4/PartitionedCallPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19226?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_19279module_wrapper_5_19281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19206?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19870

args_0
identity^
leaky_re_lu_2/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_19787

args_0
identityT
leaky_re_lu/LeakyRelu	LeakyReluargs_0*(
_output_shapes
:??????????1l
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?
?
*__inference_sequential_layer_call_fn_19009
module_wrapper_input
unknown:	d?1
	unknown_0:	?1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18993x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:?????????d
.
_user_specified_namemodule_wrapper_input
?
?
*__inference_sequential_layer_call_fn_19608

inputs
unknown:	d?1
	unknown_0:	?1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18993x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18939

args_0
identityT
leaky_re_lu/LeakyRelu	LeakyReluargs_0*(
_output_shapes
:??????????1l
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?
?
0__inference_module_wrapper_5_layer_call_fn_19920

args_0"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19206w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
g
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_19830

args_0
identityC
reshape/ShapeShapeargs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:}
reshape/ReshapeReshapeargs_0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????i
IdentityIdentityreshape/Reshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?
}
)__inference_embedding_layer_call_fn_19499

inputs
unknown:
2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_18665s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
I__inference_cond_generator_layer_call_and_return_conditional_losses_19492
latent_vecs
class_labels2
 embedding_embedding_lookup_19392:
29
'dense_tensordot_readvariableop_resource:213
%dense_biasadd_readvariableop_resource:1S
@sequential_module_wrapper_dense_1_matmul_readvariableop_resource:	d?1P
Asequential_module_wrapper_dense_1_biasadd_readvariableop_resource:	?1b
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:??L
=sequential_1_conv2d_transpose_biasadd_readvariableop_resource:	?d
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??N
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource:	?^
Csequential_1_module_wrapper_5_conv2d_conv2d_readvariableop_resource:?R
Dsequential_1_module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?embedding/embedding_lookup?8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp?7sequential/module_wrapper/dense_1/MatMul/ReadVariableOp?4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOpe
embedding/CastCastclass_labels*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding/embedding_lookupResourceGather embedding_embedding_lookup_19392embedding/Cast:y:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/19392*+
_output_shapes
:?????????2*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/19392*+
_output_shapes
:?????????2?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:21*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       s
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????1a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:1_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????1~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????1f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         |
ReshapeReshapedense/BiasAdd:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
7sequential/module_wrapper/dense_1/MatMul/ReadVariableOpReadVariableOp@sequential_module_wrapper_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0?
(sequential/module_wrapper/dense_1/MatMulMatMullatent_vecs?sequential/module_wrapper/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOpReadVariableOpAsequential_module_wrapper_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
)sequential/module_wrapper/dense_1/BiasAddBiasAdd2sequential/module_wrapper/dense_1/MatMul:product:0@sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
1sequential/module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyRelu2sequential/module_wrapper/dense_1/BiasAdd:output:0*(
_output_shapes
:??????????1?
)sequential/module_wrapper_2/reshape/ShapeShape?sequential/module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:?
7sequential/module_wrapper_2/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9sequential/module_wrapper_2/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/module_wrapper_2/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1sequential/module_wrapper_2/reshape/strided_sliceStridedSlice2sequential/module_wrapper_2/reshape/Shape:output:0@sequential/module_wrapper_2/reshape/strided_slice/stack:output:0Bsequential/module_wrapper_2/reshape/strided_slice/stack_1:output:0Bsequential/module_wrapper_2/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3sequential/module_wrapper_2/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
3sequential/module_wrapper_2/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
3sequential/module_wrapper_2/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
1sequential/module_wrapper_2/reshape/Reshape/shapePack:sequential/module_wrapper_2/reshape/strided_slice:output:0<sequential/module_wrapper_2/reshape/Reshape/shape/1:output:0<sequential/module_wrapper_2/reshape/Reshape/shape/2:output:0<sequential/module_wrapper_2/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
+sequential/module_wrapper_2/reshape/ReshapeReshape?sequential/module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0:sequential/module_wrapper_2/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV24sequential/module_wrapper_2/reshape/Reshape:output:0Reshape:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????b
#sequential_1/conv2d_transpose/ShapeShapeconcat:output:0*
T0*
_output_shapes
:{
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :g
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:}
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5sequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu	LeakyRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*0
_output_shapes
:???????????
%sequential_1/conv2d_transpose_1/ShapeShapeCsequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:}
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Csequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5sequential_1/module_wrapper_4/leaky_re_lu_2/LeakyRelu	LeakyRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*0
_output_shapes
:???????????
:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOpReadVariableOpCsequential_1_module_wrapper_5_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
+sequential_1/module_wrapper_5/conv2d/Conv2DConv2DCsequential_1/module_wrapper_4/leaky_re_lu_2/LeakyRelu:activations:0Bsequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOpReadVariableOpDsequential_1_module_wrapper_5_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
,sequential_1/module_wrapper_5/conv2d/BiasAddBiasAdd4sequential_1/module_wrapper_5/conv2d/Conv2D:output:0Csequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
)sequential_1/module_wrapper_5/conv2d/TanhTanh5sequential_1/module_wrapper_5/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
IdentityIdentity-sequential_1/module_wrapper_5/conv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup9^sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp8^sequential/module_wrapper/dense_1/MatMul/ReadVariableOp5^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp<^sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp;^sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????d:?????????: : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2t
8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp2r
7sequential/module_wrapper/dense_1/MatMul/ReadVariableOp7sequential/module_wrapper/dense_1/MatMul/ReadVariableOp2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2z
;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2x
:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:T P
'
_output_shapes
:?????????d
%
_user_specified_namelatent_vecs:UQ
'
_output_shapes
:?????????
&
_user_specified_nameclass_labels
?
?
.__inference_cond_generator_layer_call_fn_19387
latent_vecs
class_labels
unknown:
2
	unknown_0:21
	unknown_1:1
	unknown_2:	d?1
	unknown_3:	?1%
	unknown_4:??
	unknown_5:	?%
	unknown_6:??
	unknown_7:	?$
	unknown_8:?
	unknown_9:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllatent_vecsclass_labelsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_cond_generator_layer_call_and_return_conditional_losses_18774w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????d:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????d
%
_user_specified_namelatent_vecs:UQ
'
_output_shapes
:?????????
&
_user_specified_nameclass_labels
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_18993

inputs'
module_wrapper_18985:	d?1#
module_wrapper_18987:	?1
identity??&module_wrapper/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_18985module_wrapper_18987*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18964?
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18939?
 module_wrapper_2/PartitionedCallPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18923?
IdentityIdentity)module_wrapper_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????o
NoOpNoOp'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
L
0__inference_module_wrapper_2_layer_call_fn_19840

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18923i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_18894

inputs'
module_wrapper_18865:	d?1#
module_wrapper_18867:	?1
identity??&module_wrapper/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_18865module_wrapper_18867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18864?
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18875?
 module_wrapper_2/PartitionedCallPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18891?
IdentityIdentity)module_wrapper_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????o
NoOpNoOp'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
? 
?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_19020
module_wrapper_input'
module_wrapper_19012:	d?1#
module_wrapper_19014:	?1
identity??&module_wrapper/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_19012module_wrapper_19014*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18864?
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_18875?
 module_wrapper_2/PartitionedCallPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18891?
IdentityIdentity)module_wrapper_2/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????o
NoOpNoOp'^module_wrapper/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall:] Y
'
_output_shapes
:?????????d
.
_user_specified_namemodule_wrapper_input
?

?
I__inference_module_wrapper_layer_call_and_return_conditional_losses_18864

args_09
&dense_1_matmul_readvariableop_resource:	d?16
'dense_1_biasadd_readvariableop_resource:	?1
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0z
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1h
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameargs_0
?
?
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19206

args_0@
%conv2d_conv2d_readvariableop_resource:?4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
IdentityIdentityconv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?	
?
D__inference_embedding_layer_call_and_return_conditional_losses_18665

inputs(
embedding_lookup_18659:
2
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:??????????
embedding_lookupResourceGatherembedding_lookup_18659Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/18659*+
_output_shapes
:?????????2*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/18659*+
_output_shapes
:?????????2?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
0__inference_module_wrapper_3_layer_call_fn_19855

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19136i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
2__inference_conv2d_transpose_1_layer_call_fn_19119

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?D
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19710

inputsU
9conv2d_transpose_conv2d_transpose_readvariableop_resource:???
0conv2d_transpose_biasadd_readvariableop_resource:	?W
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??A
2conv2d_transpose_1_biasadd_readvariableop_resource:	?Q
6module_wrapper_5_conv2d_conv2d_readvariableop_resource:?E
7module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?-module_wrapper_5/conv2d/Conv2D/ReadVariableOpL
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :[
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
(module_wrapper_3/leaky_re_lu_1/LeakyRelu	LeakyRelu!conv2d_transpose/BiasAdd:output:0*0
_output_shapes
:??????????~
conv2d_transpose_1/ShapeShape6module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:06module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
(module_wrapper_4/leaky_re_lu_2/LeakyRelu	LeakyRelu#conv2d_transpose_1/BiasAdd:output:0*0
_output_shapes
:???????????
-module_wrapper_5/conv2d/Conv2D/ReadVariableOpReadVariableOp6module_wrapper_5_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
module_wrapper_5/conv2d/Conv2DConv2D6module_wrapper_4/leaky_re_lu_2/LeakyRelu:activations:05module_wrapper_5/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
.module_wrapper_5/conv2d/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_5_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
module_wrapper_5/conv2d/BiasAddBiasAdd'module_wrapper_5/conv2d/Conv2D:output:06module_wrapper_5/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
module_wrapper_5/conv2d/TanhTanh(module_wrapper_5/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????w
IdentityIdentity module_wrapper_5/conv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp/^module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.^module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2`
.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2^
-module_wrapper_5/conv2d/Conv2D/ReadVariableOp-module_wrapper_5/conv2d/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19850

args_0
identity^
leaky_re_lu_1/LeakyRelu	LeakyReluargs_0*0
_output_shapes
:??????????v
IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
L
0__inference_module_wrapper_3_layer_call_fn_19860

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19242i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19891

args_0@
%conv2d_conv2d_readvariableop_resource:?4
&conv2d_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
IdentityIdentityconv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
g
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_19792

args_0
identityT
leaky_re_lu/LeakyRelu	LeakyReluargs_0*(
_output_shapes
:??????????1l
IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*(
_output_shapes
:??????????1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?
?
*__inference_sequential_layer_call_fn_18901
module_wrapper_input
unknown:	d?1
	unknown_0:	?1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18894x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:?????????d
.
_user_specified_namemodule_wrapper_input
?
?
0__inference_module_wrapper_5_layer_call_fn_19911

args_0"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19161w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_19590

inputsH
5module_wrapper_dense_1_matmul_readvariableop_resource:	d?1E
6module_wrapper_dense_1_biasadd_readvariableop_resource:	?1
identity??-module_wrapper/dense_1/BiasAdd/ReadVariableOp?,module_wrapper/dense_1/MatMul/ReadVariableOp?
,module_wrapper/dense_1/MatMul/ReadVariableOpReadVariableOp5module_wrapper_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0?
module_wrapper/dense_1/MatMulMatMulinputs4module_wrapper/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
-module_wrapper/dense_1/BiasAdd/ReadVariableOpReadVariableOp6module_wrapper_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
module_wrapper/dense_1/BiasAddBiasAdd'module_wrapper/dense_1/MatMul:product:05module_wrapper/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
&module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyRelu'module_wrapper/dense_1/BiasAdd:output:0*(
_output_shapes
:??????????1?
module_wrapper_2/reshape/ShapeShape4module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:v
,module_wrapper_2/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.module_wrapper_2/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.module_wrapper_2/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&module_wrapper_2/reshape/strided_sliceStridedSlice'module_wrapper_2/reshape/Shape:output:05module_wrapper_2/reshape/strided_slice/stack:output:07module_wrapper_2/reshape/strided_slice/stack_1:output:07module_wrapper_2/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(module_wrapper_2/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :j
(module_wrapper_2/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
(module_wrapper_2/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
&module_wrapper_2/reshape/Reshape/shapePack/module_wrapper_2/reshape/strided_slice:output:01module_wrapper_2/reshape/Reshape/shape/1:output:01module_wrapper_2/reshape/Reshape/shape/2:output:01module_wrapper_2/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
 module_wrapper_2/reshape/ReshapeReshape4module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0/module_wrapper_2/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:???????????
IdentityIdentity)module_wrapper_2/reshape/Reshape:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp.^module_wrapper/dense_1/BiasAdd/ReadVariableOp-^module_wrapper/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2^
-module_wrapper/dense_1/BiasAdd/ReadVariableOp-module_wrapper/dense_1/BiasAdd/ReadVariableOp2\
,module_wrapper/dense_1/MatMul/ReadVariableOp,module_wrapper/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
g
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_18923

args_0
identityC
reshape/ShapeShapeargs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:}
reshape/ReshapeReshapeargs_0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????i
IdentityIdentityreshape/Reshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameargs_0
?z
?
I__inference_cond_generator_layer_call_and_return_conditional_losses_18774
latent_vecs
class_labels!
embedding_18666:
2
dense_18700:21
dense_18702:1S
@sequential_module_wrapper_dense_1_matmul_readvariableop_resource:	d?1P
Asequential_module_wrapper_dense_1_biasadd_readvariableop_resource:	?1b
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:??L
=sequential_1_conv2d_transpose_biasadd_readvariableop_resource:	?d
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??N
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource:	?^
Csequential_1_module_wrapper_5_conv2d_conv2d_readvariableop_resource:?R
Dsequential_1_module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp?7sequential/module_wrapper/dense_1/MatMul/ReadVariableOp?4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp?=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp?6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp??sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp?
!embedding/StatefulPartitionedCallStatefulPartitionedCallclass_labelsembedding_18666*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_18665?
dense/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0dense_18700dense_18702*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_18699f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????         ?
ReshapeReshape&dense/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:??????????
7sequential/module_wrapper/dense_1/MatMul/ReadVariableOpReadVariableOp@sequential_module_wrapper_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?1*
dtype0?
(sequential/module_wrapper/dense_1/MatMulMatMullatent_vecs?sequential/module_wrapper/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOpReadVariableOpAsequential_module_wrapper_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
)sequential/module_wrapper/dense_1/BiasAddBiasAdd2sequential/module_wrapper/dense_1/MatMul:product:0@sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
1sequential/module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyRelu2sequential/module_wrapper/dense_1/BiasAdd:output:0*(
_output_shapes
:??????????1?
)sequential/module_wrapper_2/reshape/ShapeShape?sequential/module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:?
7sequential/module_wrapper_2/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9sequential/module_wrapper_2/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/module_wrapper_2/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1sequential/module_wrapper_2/reshape/strided_sliceStridedSlice2sequential/module_wrapper_2/reshape/Shape:output:0@sequential/module_wrapper_2/reshape/strided_slice/stack:output:0Bsequential/module_wrapper_2/reshape/strided_slice/stack_1:output:0Bsequential/module_wrapper_2/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3sequential/module_wrapper_2/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
3sequential/module_wrapper_2/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :v
3sequential/module_wrapper_2/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
1sequential/module_wrapper_2/reshape/Reshape/shapePack:sequential/module_wrapper_2/reshape/strided_slice:output:0<sequential/module_wrapper_2/reshape/Reshape/shape/1:output:0<sequential/module_wrapper_2/reshape/Reshape/shape/2:output:0<sequential/module_wrapper_2/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
+sequential/module_wrapper_2/reshape/ReshapeReshape?sequential/module_wrapper_1/leaky_re_lu/LeakyRelu:activations:0:sequential/module_wrapper_2/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV24sequential/module_wrapper_2/reshape/Reshape:output:0Reshape:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????b
#sequential_1/conv2d_transpose/ShapeShapeconcat:output:0*
T0*
_output_shapes
:{
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :g
%sequential_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0.sequential_1/conv2d_transpose/stack/1:output:0.sequential_1/conv2d_transpose/stack/2:output:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:}
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5sequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu	LeakyRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*0
_output_shapes
:???????????
%sequential_1/conv2d_transpose_1/ShapeShapeCsequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:}
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :i
'sequential_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :j
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:00sequential_1/conv2d_transpose_1/stack/1:output:00sequential_1/conv2d_transpose_1/stack/2:output:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Csequential_1/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5sequential_1/module_wrapper_4/leaky_re_lu_2/LeakyRelu	LeakyRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*0
_output_shapes
:???????????
:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOpReadVariableOpCsequential_1_module_wrapper_5_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
+sequential_1/module_wrapper_5/conv2d/Conv2DConv2DCsequential_1/module_wrapper_4/leaky_re_lu_2/LeakyRelu:activations:0Bsequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOpReadVariableOpDsequential_1_module_wrapper_5_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
,sequential_1/module_wrapper_5/conv2d/BiasAddBiasAdd4sequential_1/module_wrapper_5/conv2d/Conv2D:output:0Csequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
)sequential_1/module_wrapper_5/conv2d/TanhTanh5sequential_1/module_wrapper_5/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
IdentityIdentity-sequential_1/module_wrapper_5/conv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall9^sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp8^sequential/module_wrapper/dense_1/MatMul/ReadVariableOp5^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp<^sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp;^sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????d:?????????: : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2t
8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp8sequential/module_wrapper/dense_1/BiasAdd/ReadVariableOp2r
7sequential/module_wrapper/dense_1/MatMul/ReadVariableOp7sequential/module_wrapper/dense_1/MatMul/ReadVariableOp2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2z
;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp;sequential_1/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2x
:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:sequential_1/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:T P
'
_output_shapes
:?????????d
%
_user_specified_namelatent_vecs:UQ
'
_output_shapes
:?????????
&
_user_specified_nameclass_labels
?	
?
,__inference_sequential_1_layer_call_fn_19727

inputs#
unknown:??
	unknown_0:	?%
	unknown_1:??
	unknown_2:	?$
	unknown_3:?
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19168w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
args_0/
serving_default_args_0:0?????????d
9
args_1/
serving_default_args_1:0?????????D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
emb
	emb_dense
transform_latent_vecs
	model
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?


embeddings
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
layer_with_weights-0
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
 layer_with_weights-2
 layer-4
!	variables
"trainable_variables
#regularization_losses
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
n

0
1
2
%3
&4
'5
(6
)7
*8
+9
,10"
trackable_list_wrapper
n

0
1
2
%3
&4
'5
(6
)7
*8
+9
,10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
5:3
22#cond_generator/embedding/embeddings
'

0"
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+212cond_generator/dense/kernel
':%12cond_generator/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
<_module
=	variables
>trainable_variables
?regularization_losses
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
A_module
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
F_module
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Klayers
	variables
Llayer_metrics
trainable_variables
Mlayer_regularization_losses
regularization_losses
Nmetrics
Onon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

'kernel
(bias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
T_module
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

)kernel
*bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
]_module
^	variables
_trainable_variables
`regularization_losses
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
b_module
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
 "
trackable_list_wrapper
?

glayers
!	variables
hlayer_metrics
"trainable_variables
ilayer_regularization_losses
#regularization_losses
jmetrics
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0:.	d?12module_wrapper/dense_1/kernel
*:(?12module_wrapper/dense_1/bias
3:1??2conv2d_transpose/kernel
$:"?2conv2d_transpose/bias
5:3??2conv2d_transpose_1/kernel
&:$?2conv2d_transpose_1/bias
9:7?2module_wrapper_5/conv2d/kernel
*:(2module_wrapper_5/conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

%kernel
&bias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

players
=	variables
qlayer_metrics
>trainable_variables
rlayer_regularization_losses
?regularization_losses
smetrics
tnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

ylayers
B	variables
zlayer_metrics
Ctrainable_variables
{layer_regularization_losses
Dregularization_losses
|metrics
}non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
~	variables
trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
G	variables
?layer_metrics
Htrainable_variables
 ?layer_regularization_losses
Iregularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
P	variables
?layer_metrics
Qtrainable_variables
 ?layer_regularization_losses
Rregularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
U	variables
?layer_metrics
Vtrainable_variables
 ?layer_regularization_losses
Wregularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Y	variables
?layer_metrics
Ztrainable_variables
 ?layer_regularization_losses
[regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
^	variables
?layer_metrics
_trainable_variables
 ?layer_regularization_losses
`regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

+kernel
,bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
c	variables
?layer_metrics
dtrainable_variables
 ?layer_regularization_losses
eregularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
 4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
.__inference_cond_generator_layer_call_fn_19387?
???
FullArgSpec2
args*?'
jself
jlatent_vecs
jclass_labels
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_cond_generator_layer_call_and_return_conditional_losses_19492?
???
FullArgSpec2
args*?'
jself
jlatent_vecs
jclass_labels
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
 __inference__wrapped_model_18646args_0args_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_embedding_layer_call_fn_19499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_19509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_19518?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_19548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_19569
E__inference_sequential_layer_call_and_return_conditional_losses_19590
E__inference_sequential_layer_call_and_return_conditional_losses_19020
E__inference_sequential_layer_call_and_return_conditional_losses_19031?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_sequential_layer_call_fn_18901
*__inference_sequential_layer_call_fn_19599
*__inference_sequential_layer_call_fn_19608
*__inference_sequential_layer_call_fn_19009?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19659
G__inference_sequential_1_layer_call_and_return_conditional_losses_19710
G__inference_sequential_1_layer_call_and_return_conditional_losses_19338
G__inference_sequential_1_layer_call_and_return_conditional_losses_19359?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_1_layer_call_fn_19183
,__inference_sequential_1_layer_call_fn_19727
,__inference_sequential_1_layer_call_fn_19744
,__inference_sequential_1_layer_call_fn_19317?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_18847args_0args_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_module_wrapper_layer_call_and_return_conditional_losses_19754
I__inference_module_wrapper_layer_call_and_return_conditional_losses_19764?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
.__inference_module_wrapper_layer_call_fn_19773
.__inference_module_wrapper_layer_call_fn_19782?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_19787
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_19792?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
0__inference_module_wrapper_1_layer_call_fn_19797
0__inference_module_wrapper_1_layer_call_fn_19802?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_19816
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_19830?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
0__inference_module_wrapper_2_layer_call_fn_19835
0__inference_module_wrapper_2_layer_call_fn_19840?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
0__inference_conv2d_transpose_layer_call_fn_19075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19845
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19850?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
0__inference_module_wrapper_3_layer_call_fn_19855
0__inference_module_wrapper_3_layer_call_fn_19860?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
2__inference_conv2d_transpose_1_layer_call_fn_19119?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19865
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19870?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
0__inference_module_wrapper_4_layer_call_fn_19875
0__inference_module_wrapper_4_layer_call_fn_19880?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19891
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19902?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
0__inference_module_wrapper_5_layer_call_fn_19911
0__inference_module_wrapper_5_layer_call_fn_19920?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_18646?
%&'()*+,Q?N
G?D
 ?
args_0?????????d
 ?
args_1?????????
? ";?8
6
output_1*?'
output_1??????????
I__inference_cond_generator_layer_call_and_return_conditional_losses_19492?
%&'()*+,\?Y
R?O
%?"
latent_vecs?????????d
&?#
class_labels?????????
? "-?*
#? 
0?????????
? ?
.__inference_cond_generator_layer_call_fn_19387?
%&'()*+,\?Y
R?O
%?"
latent_vecs?????????d
&?#
class_labels?????????
? " ???????????
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_19109?)*J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
2__inference_conv2d_transpose_1_layer_call_fn_19119?)*J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_19065?'(J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
0__inference_conv2d_transpose_layer_call_fn_19075?'(J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
@__inference_dense_layer_call_and_return_conditional_losses_19548d3?0
)?&
$?!
inputs?????????2
? ")?&
?
0?????????1
? ?
%__inference_dense_layer_call_fn_19518W3?0
)?&
$?!
inputs?????????2
? "??????????1?
D__inference_embedding_layer_call_and_return_conditional_losses_19509_
/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????2
? 
)__inference_embedding_layer_call_fn_19499R
/?,
%?"
 ?
inputs?????????
? "??????????2?
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_19787j@?=
&?#
!?
args_0??????????1
?

trainingp "&?#
?
0??????????1
? ?
K__inference_module_wrapper_1_layer_call_and_return_conditional_losses_19792j@?=
&?#
!?
args_0??????????1
?

trainingp"&?#
?
0??????????1
? ?
0__inference_module_wrapper_1_layer_call_fn_19797]@?=
&?#
!?
args_0??????????1
?

trainingp "???????????1?
0__inference_module_wrapper_1_layer_call_fn_19802]@?=
&?#
!?
args_0??????????1
?

trainingp"???????????1?
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_19816r@?=
&?#
!?
args_0??????????1
?

trainingp ".?+
$?!
0??????????
? ?
K__inference_module_wrapper_2_layer_call_and_return_conditional_losses_19830r@?=
&?#
!?
args_0??????????1
?

trainingp".?+
$?!
0??????????
? ?
0__inference_module_wrapper_2_layer_call_fn_19835e@?=
&?#
!?
args_0??????????1
?

trainingp "!????????????
0__inference_module_wrapper_2_layer_call_fn_19840e@?=
&?#
!?
args_0??????????1
?

trainingp"!????????????
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19845zH?E
.?+
)?&
args_0??????????
?

trainingp ".?+
$?!
0??????????
? ?
K__inference_module_wrapper_3_layer_call_and_return_conditional_losses_19850zH?E
.?+
)?&
args_0??????????
?

trainingp".?+
$?!
0??????????
? ?
0__inference_module_wrapper_3_layer_call_fn_19855mH?E
.?+
)?&
args_0??????????
?

trainingp "!????????????
0__inference_module_wrapper_3_layer_call_fn_19860mH?E
.?+
)?&
args_0??????????
?

trainingp"!????????????
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19865zH?E
.?+
)?&
args_0??????????
?

trainingp ".?+
$?!
0??????????
? ?
K__inference_module_wrapper_4_layer_call_and_return_conditional_losses_19870zH?E
.?+
)?&
args_0??????????
?

trainingp".?+
$?!
0??????????
? ?
0__inference_module_wrapper_4_layer_call_fn_19875mH?E
.?+
)?&
args_0??????????
?

trainingp "!????????????
0__inference_module_wrapper_4_layer_call_fn_19880mH?E
.?+
)?&
args_0??????????
?

trainingp"!????????????
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19891}+,H?E
.?+
)?&
args_0??????????
?

trainingp "-?*
#? 
0?????????
? ?
K__inference_module_wrapper_5_layer_call_and_return_conditional_losses_19902}+,H?E
.?+
)?&
args_0??????????
?

trainingp"-?*
#? 
0?????????
? ?
0__inference_module_wrapper_5_layer_call_fn_19911p+,H?E
.?+
)?&
args_0??????????
?

trainingp " ???????????
0__inference_module_wrapper_5_layer_call_fn_19920p+,H?E
.?+
)?&
args_0??????????
?

trainingp" ???????????
I__inference_module_wrapper_layer_call_and_return_conditional_losses_19754m%&??<
%?"
 ?
args_0?????????d
?

trainingp "&?#
?
0??????????1
? ?
I__inference_module_wrapper_layer_call_and_return_conditional_losses_19764m%&??<
%?"
 ?
args_0?????????d
?

trainingp"&?#
?
0??????????1
? ?
.__inference_module_wrapper_layer_call_fn_19773`%&??<
%?"
 ?
args_0?????????d
?

trainingp "???????????1?
.__inference_module_wrapper_layer_call_fn_19782`%&??<
%?"
 ?
args_0?????????d
?

trainingp"???????????1?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19338?'()*+,P?M
F?C
9?6
conv2d_transpose_input??????????
p 

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19359?'()*+,P?M
F?C
9?6
conv2d_transpose_input??????????
p

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19659y'()*+,@?=
6?3
)?&
inputs??????????
p 

 
? "-?*
#? 
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_19710y'()*+,@?=
6?3
)?&
inputs??????????
p

 
? "-?*
#? 
0?????????
? ?
,__inference_sequential_1_layer_call_fn_19183|'()*+,P?M
F?C
9?6
conv2d_transpose_input??????????
p 

 
? " ???????????
,__inference_sequential_1_layer_call_fn_19317|'()*+,P?M
F?C
9?6
conv2d_transpose_input??????????
p

 
? " ???????????
,__inference_sequential_1_layer_call_fn_19727l'()*+,@?=
6?3
)?&
inputs??????????
p 

 
? " ???????????
,__inference_sequential_1_layer_call_fn_19744l'()*+,@?=
6?3
)?&
inputs??????????
p

 
? " ???????????
E__inference_sequential_layer_call_and_return_conditional_losses_19020{%&E?B
;?8
.?+
module_wrapper_input?????????d
p 

 
? ".?+
$?!
0??????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_19031{%&E?B
;?8
.?+
module_wrapper_input?????????d
p

 
? ".?+
$?!
0??????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_19569m%&7?4
-?*
 ?
inputs?????????d
p 

 
? ".?+
$?!
0??????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_19590m%&7?4
-?*
 ?
inputs?????????d
p

 
? ".?+
$?!
0??????????
? ?
*__inference_sequential_layer_call_fn_18901n%&E?B
;?8
.?+
module_wrapper_input?????????d
p 

 
? "!????????????
*__inference_sequential_layer_call_fn_19009n%&E?B
;?8
.?+
module_wrapper_input?????????d
p

 
? "!????????????
*__inference_sequential_layer_call_fn_19599`%&7?4
-?*
 ?
inputs?????????d
p 

 
? "!????????????
*__inference_sequential_layer_call_fn_19608`%&7?4
-?*
 ?
inputs?????????d
p

 
? "!????????????
#__inference_signature_wrapper_18847?
%&'()*+,e?b
? 
[?X
*
args_0 ?
args_0?????????d
*
args_1 ?
args_1?????????";?8
6
output_1*?'
output_1?????????