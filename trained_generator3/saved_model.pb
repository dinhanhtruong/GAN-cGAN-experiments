??	
??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
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
module_wrapper/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??1*,
shared_namemodule_wrapper/dense/kernel
?
/module_wrapper/dense/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper/dense/kernel* 
_output_shapes
:
??1*
dtype0
?
module_wrapper/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?1**
shared_namemodule_wrapper/dense/bias
?
-module_wrapper/dense/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper/dense/bias*
_output_shapes	
:?1*
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
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?,
value?,B?, B?,
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

regularization_losses
trainable_variables
	keras_api

signatures
_
_module
	variables
regularization_losses
trainable_variables
	keras_api
_
_module
	variables
regularization_losses
trainable_variables
	keras_api
_
_module
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
_
#_module
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
_
._module
/	variables
0regularization_losses
1trainable_variables
2	keras_api
_
3_module
4	variables
5regularization_losses
6trainable_variables
7	keras_api
8
80
91
2
3
(4
)5
:6
;7
 
8
80
91
2
3
(4
)5
:6
;7
?
<metrics
=layer_regularization_losses
		variables

regularization_losses
>non_trainable_variables
trainable_variables

?layers
@layer_metrics
 
h

8kernel
9bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api

80
91
 

80
91
?
Emetrics
Flayer_regularization_losses
	variables
regularization_losses
Gnon_trainable_variables
trainable_variables

Hlayers
Ilayer_metrics
R
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
 
 
 
?
Nmetrics
Olayer_regularization_losses
	variables
regularization_losses
Pnon_trainable_variables
trainable_variables

Qlayers
Rlayer_metrics
R
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
 
 
 
?
Wmetrics
Xlayer_regularization_losses
	variables
regularization_losses
Ynon_trainable_variables
trainable_variables

Zlayers
[layer_metrics
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
\metrics
]layer_regularization_losses
	variables
 regularization_losses
^non_trainable_variables
!trainable_variables

_layers
`layer_metrics
R
a	variables
btrainable_variables
cregularization_losses
d	keras_api
 
 
 
?
emetrics
flayer_regularization_losses
$	variables
%regularization_losses
gnon_trainable_variables
&trainable_variables

hlayers
ilayer_metrics
ec
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?
jmetrics
klayer_regularization_losses
*	variables
+regularization_losses
lnon_trainable_variables
,trainable_variables

mlayers
nlayer_metrics
R
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
 
 
 
?
smetrics
tlayer_regularization_losses
/	variables
0regularization_losses
unon_trainable_variables
1trainable_variables

vlayers
wlayer_metrics
h

:kernel
;bias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api

:0
;1
 

:0
;1
?
|metrics
}layer_regularization_losses
4	variables
5regularization_losses
~non_trainable_variables
6trainable_variables

layers
?layer_metrics
WU
VARIABLE_VALUEmodule_wrapper/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEmodule_wrapper/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEmodule_wrapper_5/conv2d/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEmodule_wrapper_5/conv2d/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
8
0
1
2
3
4
5
6
7
 

80
91

80
91
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
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
J	variables
Ktrainable_variables
Lregularization_losses
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
S	variables
Ttrainable_variables
Uregularization_losses
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
a	variables
btrainable_variables
cregularization_losses
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
o	variables
ptrainable_variables
qregularization_losses
 
 
 
 
 

:0
;1

:0
;1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
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
?
$serving_default_module_wrapper_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCall$serving_default_module_wrapper_inputmodule_wrapper/dense/kernelmodule_wrapper/dense/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasmodule_wrapper_5/conv2d/kernelmodule_wrapper_5/conv2d/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_8741
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp/module_wrapper/dense/kernel/Read/ReadVariableOp-module_wrapper/dense/bias/Read/ReadVariableOp2module_wrapper_5/conv2d/kernel/Read/ReadVariableOp0module_wrapper_5/conv2d/bias/Read/ReadVariableOpConst*
Tin
2
*
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
GPU 2J 8? *&
f!R
__inference__traced_save_9143
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasmodule_wrapper/dense/kernelmodule_wrapper/dense/biasmodule_wrapper_5/conv2d/kernelmodule_wrapper_5/conv2d/bias*
Tin
2	*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_9177ݎ
?
K
/__inference_module_wrapper_2_layer_call_fn_9016

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8523i
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
?
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_9078

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
?
K
/__inference_module_wrapper_4_layer_call_fn_9051

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8400i
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
?
?
/__inference_module_wrapper_5_layer_call_fn_9096

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8462w
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
?
f
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8388

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
?
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8968

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
?
f
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8400

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
)__inference_sequential_layer_call_fn_8920

inputs
unknown:
??1
	unknown_0:	?1%
	unknown_1:??
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?$
	unknown_5:?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8622w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
1__inference_conv2d_transpose_1_layer_call_fn_8332

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
GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322?
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

?
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8564

args_08
$dense_matmul_readvariableop_resource:
??14
%dense_biasadd_readvariableop_resource:	?1
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??1*
dtype0v
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
K
/__inference_module_wrapper_1_layer_call_fn_8978

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8539a
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
"__inference_signature_wrapper_8741
module_wrapper_input
unknown:
??1
	unknown_0:	?1%
	unknown_1:??
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?$
	unknown_5:?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_8244w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
(
_output_shapes
:??????????
.
_user_specified_namemodule_wrapper_input
?\
?
D__inference_sequential_layer_call_and_return_conditional_losses_8810

inputsG
3module_wrapper_dense_matmul_readvariableop_resource:
??1C
4module_wrapper_dense_biasadd_readvariableop_resource:	?1U
9conv2d_transpose_conv2d_transpose_readvariableop_resource:???
0conv2d_transpose_biasadd_readvariableop_resource:	?W
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??A
2conv2d_transpose_1_biasadd_readvariableop_resource:	?Q
6module_wrapper_5_conv2d_conv2d_readvariableop_resource:?E
7module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?+module_wrapper/dense/BiasAdd/ReadVariableOp?*module_wrapper/dense/MatMul/ReadVariableOp?.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?-module_wrapper_5/conv2d/Conv2D/ReadVariableOp?
*module_wrapper/dense/MatMul/ReadVariableOpReadVariableOp3module_wrapper_dense_matmul_readvariableop_resource* 
_output_shapes
:
??1*
dtype0?
module_wrapper/dense/MatMulMatMulinputs2module_wrapper/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
+module_wrapper/dense/BiasAdd/ReadVariableOpReadVariableOp4module_wrapper_dense_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
module_wrapper/dense/BiasAddBiasAdd%module_wrapper/dense/MatMul:product:03module_wrapper/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
&module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyRelu%module_wrapper/dense/BiasAdd:output:0*(
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
:??????????o
conv2d_transpose/ShapeShape)module_wrapper_2/reshape/Reshape:output:0*
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
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0)module_wrapper_2/reshape/Reshape:output:0*
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
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp,^module_wrapper/dense/BiasAdd/ReadVariableOp+^module_wrapper/dense/MatMul/ReadVariableOp/^module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.^module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2Z
+module_wrapper/dense/BiasAdd/ReadVariableOp+module_wrapper/dense/BiasAdd/ReadVariableOp2X
*module_wrapper/dense/MatMul/ReadVariableOp*module_wrapper/dense/MatMul/ReadVariableOp2`
.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2^
-module_wrapper_5/conv2d/Conv2D/ReadVariableOp-module_wrapper_5/conv2d/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_8420

inputs'
module_wrapper_8350:
??1"
module_wrapper_8352:	?11
conv2d_transpose_8378:??$
conv2d_transpose_8380:	?3
conv2d_transpose_1_8390:??&
conv2d_transpose_1_8392:	?0
module_wrapper_5_8414:?#
module_wrapper_5_8416:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?&module_wrapper/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_8350module_wrapper_8352*
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
GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8349?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8360?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8376?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_2/PartitionedCall:output:0conv2d_transpose_8378conv2d_transpose_8380*
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
GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8388?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_8390conv2d_transpose_1_8392*
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
GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8400?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_8414module_wrapper_5_8416*
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8413?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8992

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
?
f
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8376

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
?
f
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_9021

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
?
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8413

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
)__inference_sequential_layer_call_fn_8439
module_wrapper_input
unknown:
??1
	unknown_0:	?1%
	unknown_1:??
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?$
	unknown_5:?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8420w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
(
_output_shapes
:??????????
.
_user_specified_namemodule_wrapper_input
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_8690
module_wrapper_input'
module_wrapper_8665:
??1"
module_wrapper_8667:	?11
conv2d_transpose_8672:??$
conv2d_transpose_8674:	?3
conv2d_transpose_1_8678:??&
conv2d_transpose_1_8680:	?0
module_wrapper_5_8684:?#
module_wrapper_5_8686:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?&module_wrapper/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_8665module_wrapper_8667*
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
GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8349?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8360?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8376?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_2/PartitionedCall:output:0conv2d_transpose_8672conv2d_transpose_8674*
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
GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8388?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_8678conv2d_transpose_1_8680*
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
GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8400?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_8684module_wrapper_5_8686*
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8413?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:^ Z
(
_output_shapes
:??????????
.
_user_specified_namemodule_wrapper_input
?
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8963

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
?j
?	
__inference__wrapped_model_8244
module_wrapper_inputR
>sequential_module_wrapper_dense_matmul_readvariableop_resource:
??1N
?sequential_module_wrapper_dense_biasadd_readvariableop_resource:	?1`
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource:??J
;sequential_conv2d_transpose_biasadd_readvariableop_resource:	?b
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??L
=sequential_conv2d_transpose_1_biasadd_readvariableop_resource:	?\
Asequential_module_wrapper_5_conv2d_conv2d_readvariableop_resource:?P
Bsequential_module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??2sequential/conv2d_transpose/BiasAdd/ReadVariableOp?;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp?4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp?=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?6sequential/module_wrapper/dense/BiasAdd/ReadVariableOp?5sequential/module_wrapper/dense/MatMul/ReadVariableOp?9sequential/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?8sequential/module_wrapper_5/conv2d/Conv2D/ReadVariableOp?
5sequential/module_wrapper/dense/MatMul/ReadVariableOpReadVariableOp>sequential_module_wrapper_dense_matmul_readvariableop_resource* 
_output_shapes
:
??1*
dtype0?
&sequential/module_wrapper/dense/MatMulMatMulmodule_wrapper_input=sequential/module_wrapper/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
6sequential/module_wrapper/dense/BiasAdd/ReadVariableOpReadVariableOp?sequential_module_wrapper_dense_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
'sequential/module_wrapper/dense/BiasAddBiasAdd0sequential/module_wrapper/dense/MatMul:product:0>sequential/module_wrapper/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
1sequential/module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyRelu0sequential/module_wrapper/dense/BiasAdd:output:0*(
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
:???????????
!sequential/conv2d_transpose/ShapeShape4sequential/module_wrapper_2/reshape/Reshape:output:0*
T0*
_output_shapes
:y
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0,sequential/conv2d_transpose/stack/1:output:0,sequential/conv2d_transpose/stack/2:output:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:{
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:04sequential/module_wrapper_2/reshape/Reshape:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
2sequential/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp;sequential_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#sequential/conv2d_transpose/BiasAddBiasAdd5sequential/conv2d_transpose/conv2d_transpose:output:0:sequential/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
3sequential/module_wrapper_3/leaky_re_lu_1/LeakyRelu	LeakyRelu,sequential/conv2d_transpose/BiasAdd:output:0*0
_output_shapes
:???????????
#sequential/conv2d_transpose_1/ShapeShapeAsequential/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:{
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :g
%sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :h
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0.sequential/conv2d_transpose_1/stack/1:output:0.sequential/conv2d_transpose_1/stack/2:output:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:}
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Asequential/module_wrapper_3/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
%sequential/conv2d_transpose_1/BiasAddBiasAdd7sequential/conv2d_transpose_1/conv2d_transpose:output:0<sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
3sequential/module_wrapper_4/leaky_re_lu_2/LeakyRelu	LeakyRelu.sequential/conv2d_transpose_1/BiasAdd:output:0*0
_output_shapes
:???????????
8sequential/module_wrapper_5/conv2d/Conv2D/ReadVariableOpReadVariableOpAsequential_module_wrapper_5_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
)sequential/module_wrapper_5/conv2d/Conv2DConv2DAsequential/module_wrapper_4/leaky_re_lu_2/LeakyRelu:activations:0@sequential/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
9sequential/module_wrapper_5/conv2d/BiasAdd/ReadVariableOpReadVariableOpBsequential_module_wrapper_5_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
*sequential/module_wrapper_5/conv2d/BiasAddBiasAdd2sequential/module_wrapper_5/conv2d/Conv2D:output:0Asequential/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
'sequential/module_wrapper_5/conv2d/TanhTanh3sequential/module_wrapper_5/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
IdentityIdentity+sequential/module_wrapper_5/conv2d/Tanh:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp3^sequential/conv2d_transpose/BiasAdd/ReadVariableOp<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp5^sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential/module_wrapper/dense/BiasAdd/ReadVariableOp6^sequential/module_wrapper/dense/MatMul/ReadVariableOp:^sequential/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp9^sequential/module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2h
2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2sequential/conv2d_transpose/BiasAdd/ReadVariableOp2z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2l
4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp4sequential/conv2d_transpose_1/BiasAdd/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential/module_wrapper/dense/BiasAdd/ReadVariableOp6sequential/module_wrapper/dense/BiasAdd/ReadVariableOp2n
5sequential/module_wrapper/dense/MatMul/ReadVariableOp5sequential/module_wrapper/dense/MatMul/ReadVariableOp2v
9sequential/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp9sequential/module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2t
8sequential/module_wrapper_5/conv2d/Conv2D/ReadVariableOp8sequential/module_wrapper_5/conv2d/Conv2D/ReadVariableOp:^ Z
(
_output_shapes
:??????????
.
_user_specified_namemodule_wrapper_input
?
?
/__inference_module_wrapper_5_layer_call_fn_9087

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8413w
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
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_8718
module_wrapper_input'
module_wrapper_8693:
??1"
module_wrapper_8695:	?11
conv2d_transpose_8700:??$
conv2d_transpose_8702:	?3
conv2d_transpose_1_8706:??&
conv2d_transpose_1_8708:	?0
module_wrapper_5_8712:?#
module_wrapper_5_8714:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?&module_wrapper/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_8693module_wrapper_8695*
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
GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8564?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8539?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8523?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_2/PartitionedCall:output:0conv2d_transpose_8700conv2d_transpose_8702*
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
GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8498?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_8706conv2d_transpose_1_8708*
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
GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8482?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_8712module_wrapper_5_8714*
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8462?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:^ Z
(
_output_shapes
:??????????
.
_user_specified_namemodule_wrapper_input
?
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8360

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
?
f
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8539

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
?
K
/__inference_module_wrapper_4_layer_call_fn_9056

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8482i
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
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278

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
?
K
/__inference_module_wrapper_2_layer_call_fn_9011

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8376i
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
?
f
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_9046

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
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8940

args_08
$dense_matmul_readvariableop_resource:
??14
%dense_biasadd_readvariableop_resource:	?1
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??1*
dtype0v
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
/__inference_conv2d_transpose_layer_call_fn_8288

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
GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278?
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
?
f
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_9041

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
?
K
/__inference_module_wrapper_1_layer_call_fn_8973

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8360a
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
?
__inference__traced_save_9143
file_prefix6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop:
6savev2_module_wrapper_dense_kernel_read_readvariableop8
4savev2_module_wrapper_dense_bias_read_readvariableop=
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
:	*
dtype0*?
value?B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop6savev2_module_wrapper_dense_kernel_read_readvariableop4savev2_module_wrapper_dense_bias_read_readvariableop9savev2_module_wrapper_5_conv2d_kernel_read_readvariableop7savev2_module_wrapper_5_conv2d_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	?
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

identity_1Identity_1:output:0*y
_input_shapesh
f: :??:?:??:?:
??1:?1:?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??1:!

_output_shapes	
:?1:-)
'
_output_shapes
:?: 

_output_shapes
::	

_output_shapes
: 
?
f
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8482

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
f
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_9006

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
K
/__inference_module_wrapper_3_layer_call_fn_9031

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8388i
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
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8462

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
)__inference_sequential_layer_call_fn_8662
module_wrapper_input
unknown:
??1
	unknown_0:	?1%
	unknown_1:??
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?$
	unknown_5:?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8622w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
(
_output_shapes
:??????????
.
_user_specified_namemodule_wrapper_input
?
?
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_9067

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
?%
?
 __inference__traced_restore_9177
file_prefixD
(assignvariableop_conv2d_transpose_kernel:??7
(assignvariableop_1_conv2d_transpose_bias:	?H
,assignvariableop_2_conv2d_transpose_1_kernel:??9
*assignvariableop_3_conv2d_transpose_1_bias:	?B
.assignvariableop_4_module_wrapper_dense_kernel:
??1;
,assignvariableop_5_module_wrapper_dense_bias:	?1L
1assignvariableop_6_module_wrapper_5_conv2d_kernel:?=
/assignvariableop_7_module_wrapper_5_conv2d_bias:

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp(assignvariableop_conv2d_transpose_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_conv2d_transpose_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_conv2d_transpose_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv2d_transpose_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_module_wrapper_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_module_wrapper_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp1assignvariableop_6_module_wrapper_5_conv2d_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_module_wrapper_5_conv2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8349

args_08
$dense_matmul_readvariableop_resource:
??14
%dense_biasadd_readvariableop_resource:	?1
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??1*
dtype0v
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
?
-__inference_module_wrapper_layer_call_fn_8949

args_0
unknown:
??1
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
GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8349p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?

?
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8930

args_08
$dense_matmul_readvariableop_resource:
??14
%dense_biasadd_readvariableop_resource:	?1
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??1*
dtype0v
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1f
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????1?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0
?
f
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8523

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
? 
?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322

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
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_8622

inputs'
module_wrapper_8597:
??1"
module_wrapper_8599:	?11
conv2d_transpose_8604:??$
conv2d_transpose_8606:	?3
conv2d_transpose_1_8610:??&
conv2d_transpose_1_8612:	?0
module_wrapper_5_8616:?#
module_wrapper_5_8618:
identity??(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?&module_wrapper/StatefulPartitionedCall?(module_wrapper_5/StatefulPartitionedCall?
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_8597module_wrapper_8599*
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
GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8564?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8539?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8523?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_2/PartitionedCall:output:0conv2d_transpose_8604conv2d_transpose_8606*
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
GPU 2J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8498?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_3/PartitionedCall:output:0conv2d_transpose_1_8610conv2d_transpose_1_8612*
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
GPU 2J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322?
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_8482?
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0module_wrapper_5_8616module_wrapper_5_8618*
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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_8462?
IdentityIdentity1module_wrapper_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall'^module_wrapper/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_module_wrapper_3_layer_call_fn_9036

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
GPU 2J 8? *S
fNRL
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8498i
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
?\
?
D__inference_sequential_layer_call_and_return_conditional_losses_8878

inputsG
3module_wrapper_dense_matmul_readvariableop_resource:
??1C
4module_wrapper_dense_biasadd_readvariableop_resource:	?1U
9conv2d_transpose_conv2d_transpose_readvariableop_resource:???
0conv2d_transpose_biasadd_readvariableop_resource:	?W
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:??A
2conv2d_transpose_1_biasadd_readvariableop_resource:	?Q
6module_wrapper_5_conv2d_conv2d_readvariableop_resource:?E
7module_wrapper_5_conv2d_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?+module_wrapper/dense/BiasAdd/ReadVariableOp?*module_wrapper/dense/MatMul/ReadVariableOp?.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp?-module_wrapper_5/conv2d/Conv2D/ReadVariableOp?
*module_wrapper/dense/MatMul/ReadVariableOpReadVariableOp3module_wrapper_dense_matmul_readvariableop_resource* 
_output_shapes
:
??1*
dtype0?
module_wrapper/dense/MatMulMatMulinputs2module_wrapper/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
+module_wrapper/dense/BiasAdd/ReadVariableOpReadVariableOp4module_wrapper_dense_biasadd_readvariableop_resource*
_output_shapes	
:?1*
dtype0?
module_wrapper/dense/BiasAddBiasAdd%module_wrapper/dense/MatMul:product:03module_wrapper/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????1?
&module_wrapper_1/leaky_re_lu/LeakyRelu	LeakyRelu%module_wrapper/dense/BiasAdd:output:0*(
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
:??????????o
conv2d_transpose/ShapeShape)module_wrapper_2/reshape/Reshape:output:0*
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
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0)module_wrapper_2/reshape/Reshape:output:0*
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
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp,^module_wrapper/dense/BiasAdd/ReadVariableOp+^module_wrapper/dense/MatMul/ReadVariableOp/^module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.^module_wrapper_5/conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2Z
+module_wrapper/dense/BiasAdd/ReadVariableOp+module_wrapper/dense/BiasAdd/ReadVariableOp2X
*module_wrapper/dense/MatMul/ReadVariableOp*module_wrapper/dense/MatMul/ReadVariableOp2`
.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp.module_wrapper_5/conv2d/BiasAdd/ReadVariableOp2^
-module_wrapper_5/conv2d/Conv2D/ReadVariableOp-module_wrapper_5/conv2d/Conv2D/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_sequential_layer_call_fn_8899

inputs
unknown:
??1
	unknown_0:	?1%
	unknown_1:??
	unknown_2:	?%
	unknown_3:??
	unknown_4:	?$
	unknown_5:?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_8420w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_8498

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
?
f
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_9026

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
?
?
-__inference_module_wrapper_layer_call_fn_8958

args_0
unknown:
??1
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
GPU 2J 8? *Q
fLRJ
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8564p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameargs_0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
V
module_wrapper_input>
&serving_default_module_wrapper_input:0??????????L
module_wrapper_58
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		variables

regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
_module
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
_module
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
_module
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
#_module
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
._module
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
3_module
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
X
80
91
2
3
(4
)5
:6
;7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
80
91
2
3
(4
)5
:6
;7"
trackable_list_wrapper
?
<metrics
=layer_regularization_losses
		variables

regularization_losses
>non_trainable_variables
trainable_variables

?layers
@layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

8kernel
9bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
Emetrics
Flayer_regularization_losses
	variables
regularization_losses
Gnon_trainable_variables
trainable_variables

Hlayers
Ilayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
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
Nmetrics
Olayer_regularization_losses
	variables
regularization_losses
Pnon_trainable_variables
trainable_variables

Qlayers
Rlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
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
Wmetrics
Xlayer_regularization_losses
	variables
regularization_losses
Ynon_trainable_variables
trainable_variables

Zlayers
[layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv2d_transpose/kernel
$:"?2conv2d_transpose/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
\metrics
]layer_regularization_losses
	variables
 regularization_losses
^non_trainable_variables
!trainable_variables

_layers
`layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
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
emetrics
flayer_regularization_losses
$	variables
%regularization_losses
gnon_trainable_variables
&trainable_variables

hlayers
ilayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3??2conv2d_transpose_1/kernel
&:$?2conv2d_transpose_1/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
jmetrics
klayer_regularization_losses
*	variables
+regularization_losses
lnon_trainable_variables
,trainable_variables

mlayers
nlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
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
smetrics
tlayer_regularization_losses
/	variables
0regularization_losses
unon_trainable_variables
1trainable_variables

vlayers
wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

:kernel
;bias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
|metrics
}layer_regularization_losses
4	variables
5regularization_losses
~non_trainable_variables
6trainable_variables

layers
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-
??12module_wrapper/dense/kernel
(:&?12module_wrapper/dense/bias
9:7?2module_wrapper_5/conv2d/kernel
*:(2module_wrapper_5/conv2d/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
__inference__wrapped_model_8244?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *4?1
/?,
module_wrapper_input??????????
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_8810
D__inference_sequential_layer_call_and_return_conditional_losses_8878
D__inference_sequential_layer_call_and_return_conditional_losses_8690
D__inference_sequential_layer_call_and_return_conditional_losses_8718?
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
)__inference_sequential_layer_call_fn_8439
)__inference_sequential_layer_call_fn_8899
)__inference_sequential_layer_call_fn_8920
)__inference_sequential_layer_call_fn_8662?
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
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8930
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8940?
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
-__inference_module_wrapper_layer_call_fn_8949
-__inference_module_wrapper_layer_call_fn_8958?
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
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8963
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8968?
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
/__inference_module_wrapper_1_layer_call_fn_8973
/__inference_module_wrapper_1_layer_call_fn_8978?
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
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8992
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_9006?
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
/__inference_module_wrapper_2_layer_call_fn_9011
/__inference_module_wrapper_2_layer_call_fn_9016?
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
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278?
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
/__inference_conv2d_transpose_layer_call_fn_8288?
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
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_9021
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_9026?
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
/__inference_module_wrapper_3_layer_call_fn_9031
/__inference_module_wrapper_3_layer_call_fn_9036?
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
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322?
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
1__inference_conv2d_transpose_1_layer_call_fn_8332?
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
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_9041
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_9046?
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
/__inference_module_wrapper_4_layer_call_fn_9051
/__inference_module_wrapper_4_layer_call_fn_9056?
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
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_9067
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_9078?
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
/__inference_module_wrapper_5_layer_call_fn_9087
/__inference_module_wrapper_5_layer_call_fn_9096?
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
?B?
"__inference_signature_wrapper_8741module_wrapper_input"?
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
__inference__wrapped_model_8244?89():;>?;
4?1
/?,
module_wrapper_input??????????
? "K?H
F
module_wrapper_52?/
module_wrapper_5??????????
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_8322?()J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
1__inference_conv2d_transpose_1_layer_call_fn_8332?()J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_8278?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
/__inference_conv2d_transpose_layer_call_fn_8288?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8963j@?=
&?#
!?
args_0??????????1
?

trainingp "&?#
?
0??????????1
? ?
J__inference_module_wrapper_1_layer_call_and_return_conditional_losses_8968j@?=
&?#
!?
args_0??????????1
?

trainingp"&?#
?
0??????????1
? ?
/__inference_module_wrapper_1_layer_call_fn_8973]@?=
&?#
!?
args_0??????????1
?

trainingp "???????????1?
/__inference_module_wrapper_1_layer_call_fn_8978]@?=
&?#
!?
args_0??????????1
?

trainingp"???????????1?
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_8992r@?=
&?#
!?
args_0??????????1
?

trainingp ".?+
$?!
0??????????
? ?
J__inference_module_wrapper_2_layer_call_and_return_conditional_losses_9006r@?=
&?#
!?
args_0??????????1
?

trainingp".?+
$?!
0??????????
? ?
/__inference_module_wrapper_2_layer_call_fn_9011e@?=
&?#
!?
args_0??????????1
?

trainingp "!????????????
/__inference_module_wrapper_2_layer_call_fn_9016e@?=
&?#
!?
args_0??????????1
?

trainingp"!????????????
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_9021zH?E
.?+
)?&
args_0??????????
?

trainingp ".?+
$?!
0??????????
? ?
J__inference_module_wrapper_3_layer_call_and_return_conditional_losses_9026zH?E
.?+
)?&
args_0??????????
?

trainingp".?+
$?!
0??????????
? ?
/__inference_module_wrapper_3_layer_call_fn_9031mH?E
.?+
)?&
args_0??????????
?

trainingp "!????????????
/__inference_module_wrapper_3_layer_call_fn_9036mH?E
.?+
)?&
args_0??????????
?

trainingp"!????????????
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_9041zH?E
.?+
)?&
args_0??????????
?

trainingp ".?+
$?!
0??????????
? ?
J__inference_module_wrapper_4_layer_call_and_return_conditional_losses_9046zH?E
.?+
)?&
args_0??????????
?

trainingp".?+
$?!
0??????????
? ?
/__inference_module_wrapper_4_layer_call_fn_9051mH?E
.?+
)?&
args_0??????????
?

trainingp "!????????????
/__inference_module_wrapper_4_layer_call_fn_9056mH?E
.?+
)?&
args_0??????????
?

trainingp"!????????????
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_9067}:;H?E
.?+
)?&
args_0??????????
?

trainingp "-?*
#? 
0?????????
? ?
J__inference_module_wrapper_5_layer_call_and_return_conditional_losses_9078}:;H?E
.?+
)?&
args_0??????????
?

trainingp"-?*
#? 
0?????????
? ?
/__inference_module_wrapper_5_layer_call_fn_9087p:;H?E
.?+
)?&
args_0??????????
?

trainingp " ???????????
/__inference_module_wrapper_5_layer_call_fn_9096p:;H?E
.?+
)?&
args_0??????????
?

trainingp" ???????????
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8930n89@?=
&?#
!?
args_0??????????
?

trainingp "&?#
?
0??????????1
? ?
H__inference_module_wrapper_layer_call_and_return_conditional_losses_8940n89@?=
&?#
!?
args_0??????????
?

trainingp"&?#
?
0??????????1
? ?
-__inference_module_wrapper_layer_call_fn_8949a89@?=
&?#
!?
args_0??????????
?

trainingp "???????????1?
-__inference_module_wrapper_layer_call_fn_8958a89@?=
&?#
!?
args_0??????????
?

trainingp"???????????1?
D__inference_sequential_layer_call_and_return_conditional_losses_8690?89():;F?C
<?9
/?,
module_wrapper_input??????????
p 

 
? "-?*
#? 
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_8718?89():;F?C
<?9
/?,
module_wrapper_input??????????
p

 
? "-?*
#? 
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_8810s89():;8?5
.?+
!?
inputs??????????
p 

 
? "-?*
#? 
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_8878s89():;8?5
.?+
!?
inputs??????????
p

 
? "-?*
#? 
0?????????
? ?
)__inference_sequential_layer_call_fn_8439t89():;F?C
<?9
/?,
module_wrapper_input??????????
p 

 
? " ???????????
)__inference_sequential_layer_call_fn_8662t89():;F?C
<?9
/?,
module_wrapper_input??????????
p

 
? " ???????????
)__inference_sequential_layer_call_fn_8899f89():;8?5
.?+
!?
inputs??????????
p 

 
? " ???????????
)__inference_sequential_layer_call_fn_8920f89():;8?5
.?+
!?
inputs??????????
p

 
? " ???????????
"__inference_signature_wrapper_8741?89():;V?S
? 
L?I
G
module_wrapper_input/?,
module_wrapper_input??????????"K?H
F
module_wrapper_52?/
module_wrapper_5?????????