пу
Аѕ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
∞
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.11.02v2.11.0-rc2-15-g6290819256d8ШЮ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
А
Adam/v/dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_57/bias
y
(Adam/v/dense_57/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_57/bias*
_output_shapes
:*
dtype0
А
Adam/m/dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_57/bias
y
(Adam/m/dense_57/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_57/bias*
_output_shapes
:*
dtype0
И
Adam/v/dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_57/kernel
Б
*Adam/v/dense_57/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_57/kernel*
_output_shapes

:*
dtype0
И
Adam/m/dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_57/kernel
Б
*Adam/m/dense_57/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_57/kernel*
_output_shapes

:*
dtype0
Ш
 Adam/v/lstm_75/lstm_cell_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/v/lstm_75/lstm_cell_76/bias
С
4Adam/v/lstm_75/lstm_cell_76/bias/Read/ReadVariableOpReadVariableOp Adam/v/lstm_75/lstm_cell_76/bias*
_output_shapes
:*
dtype0
Ш
 Adam/m/lstm_75/lstm_cell_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/m/lstm_75/lstm_cell_76/bias
С
4Adam/m/lstm_75/lstm_cell_76/bias/Read/ReadVariableOpReadVariableOp Adam/m/lstm_75/lstm_cell_76/bias*
_output_shapes
:*
dtype0
і
,Adam/v/lstm_75/lstm_cell_76/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/v/lstm_75/lstm_cell_76/recurrent_kernel
≠
@Adam/v/lstm_75/lstm_cell_76/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/v/lstm_75/lstm_cell_76/recurrent_kernel*
_output_shapes

:*
dtype0
і
,Adam/m/lstm_75/lstm_cell_76/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/m/lstm_75/lstm_cell_76/recurrent_kernel
≠
@Adam/m/lstm_75/lstm_cell_76/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/m/lstm_75/lstm_cell_76/recurrent_kernel*
_output_shapes

:*
dtype0
†
"Adam/v/lstm_75/lstm_cell_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/v/lstm_75/lstm_cell_76/kernel
Щ
6Adam/v/lstm_75/lstm_cell_76/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_75/lstm_cell_76/kernel*
_output_shapes

:*
dtype0
†
"Adam/m/lstm_75/lstm_cell_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/m/lstm_75/lstm_cell_76/kernel
Щ
6Adam/m/lstm_75/lstm_cell_76/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_75/lstm_cell_76/kernel*
_output_shapes

:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
К
lstm_75/lstm_cell_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_75/lstm_cell_76/bias
Г
-lstm_75/lstm_cell_76/bias/Read/ReadVariableOpReadVariableOplstm_75/lstm_cell_76/bias*
_output_shapes
:*
dtype0
¶
%lstm_75/lstm_cell_76/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%lstm_75/lstm_cell_76/recurrent_kernel
Я
9lstm_75/lstm_cell_76/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_75/lstm_cell_76/recurrent_kernel*
_output_shapes

:*
dtype0
Т
lstm_75/lstm_cell_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namelstm_75/lstm_cell_76/kernel
Л
/lstm_75/lstm_cell_76/kernel/Read/ReadVariableOpReadVariableOplstm_75/lstm_cell_76/kernel*
_output_shapes

:*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:*
dtype0
z
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_57/kernel
s
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes

:*
dtype0
И
serving_default_lstm_75_inputPlaceholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
√
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_75_inputlstm_75/lstm_cell_76/kernel%lstm_75/lstm_cell_76/recurrent_kernellstm_75/lstm_cell_76/biasdense_57/kerneldense_57/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_22142087

NoOpNoOp
к/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*•/
valueЫ/BШ/ BС/
І
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
•
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
¶
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
'
%0
&1
'2
#3
$4*
'
%0
&1
'2
#3
$4*
* 
∞
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
-trace_0
.trace_1
/trace_2
0trace_3* 
6
1trace_0
2trace_1
3trace_2
4trace_3* 
* 
Б
5
_variables
6_iterations
7_learning_rate
8_index_dict
9
_momentums
:_velocities
;_update_step_xla*

<serving_default* 

%0
&1
'2*

%0
&1
'2*
* 
Я

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_3* 
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
* 
г
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
Q_random_generator
R
state_size

%kernel
&recurrent_kernel
'bias*
* 
* 
* 
* 
С
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Xtrace_0
Ytrace_1* 

Ztrace_0
[trace_1* 
* 

#0
$1*

#0
$1*
* 
У
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

atrace_0* 

btrace_0* 
_Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_57/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_75/lstm_cell_76/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_75/lstm_cell_76/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_75/lstm_cell_76/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

c0
d1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
R
60
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
e0
g1
i2
k3
m4*
'
f0
h1
j2
l3
n4*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1
'2*

%0
&1
'2*
* 
У
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

ttrace_0
utrace_1* 

vtrace_0
wtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
x	variables
y	keras_api
	ztotal
	{count*
I
|	variables
}	keras_api
	~total
	count
А
_fn_kwargs*
mg
VARIABLE_VALUE"Adam/m/lstm_75/lstm_cell_76/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_75/lstm_cell_76/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/m/lstm_75/lstm_cell_76/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/v/lstm_75/lstm_cell_76/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_75/lstm_cell_76/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_75/lstm_cell_76/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_57/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_57/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_57/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_57/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

z0
{1*

x	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

~0
1*

|	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
”	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp/lstm_75/lstm_cell_76/kernel/Read/ReadVariableOp9lstm_75/lstm_cell_76/recurrent_kernel/Read/ReadVariableOp-lstm_75/lstm_cell_76/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/lstm_75/lstm_cell_76/kernel/Read/ReadVariableOp6Adam/v/lstm_75/lstm_cell_76/kernel/Read/ReadVariableOp@Adam/m/lstm_75/lstm_cell_76/recurrent_kernel/Read/ReadVariableOp@Adam/v/lstm_75/lstm_cell_76/recurrent_kernel/Read/ReadVariableOp4Adam/m/lstm_75/lstm_cell_76/bias/Read/ReadVariableOp4Adam/v/lstm_75/lstm_cell_76/bias/Read/ReadVariableOp*Adam/m/dense_57/kernel/Read/ReadVariableOp*Adam/v/dense_57/kernel/Read/ReadVariableOp(Adam/m/dense_57/bias/Read/ReadVariableOp(Adam/v/dense_57/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*"
Tin
2	*
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
GPU 2J 8В **
f%R#
!__inference__traced_save_22143282
™
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_57/kerneldense_57/biaslstm_75/lstm_cell_76/kernel%lstm_75/lstm_cell_76/recurrent_kernellstm_75/lstm_cell_76/bias	iterationlearning_rate"Adam/m/lstm_75/lstm_cell_76/kernel"Adam/v/lstm_75/lstm_cell_76/kernel,Adam/m/lstm_75/lstm_cell_76/recurrent_kernel,Adam/v/lstm_75/lstm_cell_76/recurrent_kernel Adam/m/lstm_75/lstm_cell_76/bias Adam/v/lstm_75/lstm_cell_76/biasAdam/m/dense_57/kernelAdam/v/dense_57/kernelAdam/m/dense_57/biasAdam/v/dense_57/biastotal_1count_1totalcount*!
Tin
2*
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_22143355ј≠
ƒ]
О
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142269

inputsE
3lstm_75_lstm_cell_76_matmul_readvariableop_resource:G
5lstm_75_lstm_cell_76_matmul_1_readvariableop_resource:B
4lstm_75_lstm_cell_76_biasadd_readvariableop_resource:9
'dense_57_matmul_readvariableop_resource:6
(dense_57_biasadd_readvariableop_resource:
identityИҐdense_57/BiasAdd/ReadVariableOpҐdense_57/MatMul/ReadVariableOpҐ+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOpҐ*lstm_75/lstm_cell_76/MatMul/ReadVariableOpҐ,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOpҐlstm_75/whileC
lstm_75/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_75/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_75/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_75/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_75/strided_sliceStridedSlicelstm_75/Shape:output:0$lstm_75/strided_slice/stack:output:0&lstm_75/strided_slice/stack_1:output:0&lstm_75/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_75/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Л
lstm_75/zeros/packedPacklstm_75/strided_slice:output:0lstm_75/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_75/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
lstm_75/zerosFilllstm_75/zeros/packed:output:0lstm_75/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
lstm_75/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :П
lstm_75/zeros_1/packedPacklstm_75/strided_slice:output:0!lstm_75/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_75/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
lstm_75/zeros_1Filllstm_75/zeros_1/packed:output:0lstm_75/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
lstm_75/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_75/transpose	Transposeinputslstm_75/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€T
lstm_75/Shape_1Shapelstm_75/transpose:y:0*
T0*
_output_shapes
:g
lstm_75/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_75/strided_slice_1StridedSlicelstm_75/Shape_1:output:0&lstm_75/strided_slice_1/stack:output:0(lstm_75/strided_slice_1/stack_1:output:0(lstm_75/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_75/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ћ
lstm_75/TensorArrayV2TensorListReserve,lstm_75/TensorArrayV2/element_shape:output:0 lstm_75/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“О
=lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ш
/lstm_75/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_75/transpose:y:0Flstm_75/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“g
lstm_75/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
lstm_75/strided_slice_2StridedSlicelstm_75/transpose:y:0&lstm_75/strided_slice_2/stack:output:0(lstm_75/strided_slice_2/stack_1:output:0(lstm_75/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЮ
*lstm_75/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3lstm_75_lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0≠
lstm_75/lstm_cell_76/MatMulMatMul lstm_75/strided_slice_2:output:02lstm_75/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5lstm_75_lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0І
lstm_75/lstm_cell_76/MatMul_1MatMullstm_75/zeros:output:04lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€£
lstm_75/lstm_cell_76/addAddV2%lstm_75/lstm_cell_76/MatMul:product:0'lstm_75/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4lstm_75_lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
lstm_75/lstm_cell_76/BiasAddBiasAddlstm_75/lstm_cell_76/add:z:03lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
$lstm_75/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :х
lstm_75/lstm_cell_76/splitSplit-lstm_75/lstm_cell_76/split/split_dim:output:0%lstm_75/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_split~
lstm_75/lstm_cell_76/SigmoidSigmoid#lstm_75/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_75/lstm_cell_76/Sigmoid_1Sigmoid#lstm_75/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
lstm_75/lstm_cell_76/mulMul"lstm_75/lstm_cell_76/Sigmoid_1:y:0lstm_75/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_75/lstm_cell_76/ReluRelu#lstm_75/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm_75/lstm_cell_76/mul_1Mul lstm_75/lstm_cell_76/Sigmoid:y:0'lstm_75/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€У
lstm_75/lstm_cell_76/add_1AddV2lstm_75/lstm_cell_76/mul:z:0lstm_75/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_75/lstm_cell_76/Sigmoid_2Sigmoid#lstm_75/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€u
lstm_75/lstm_cell_76/Relu_1Relulstm_75/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
lstm_75/lstm_cell_76/mul_2Mul"lstm_75/lstm_cell_76/Sigmoid_2:y:0)lstm_75/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€v
%lstm_75/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   f
$lstm_75/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_75/TensorArrayV2_1TensorListReserve.lstm_75/TensorArrayV2_1/element_shape:output:0-lstm_75/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“N
lstm_75/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_75/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€\
lstm_75/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ц
lstm_75/whileWhile#lstm_75/while/loop_counter:output:0)lstm_75/while/maximum_iterations:output:0lstm_75/time:output:0 lstm_75/TensorArrayV2_1:handle:0lstm_75/zeros:output:0lstm_75/zeros_1:output:0 lstm_75/strided_slice_1:output:0?lstm_75/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_75_lstm_cell_76_matmul_readvariableop_resource5lstm_75_lstm_cell_76_matmul_1_readvariableop_resource4lstm_75_lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_75_while_body_22142177*'
condR
lstm_75_while_cond_22142176*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Й
8lstm_75/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   о
*lstm_75/TensorArrayV2Stack/TensorListStackTensorListStacklstm_75/while:output:3Alstm_75/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsp
lstm_75/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€i
lstm_75/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
lstm_75/strided_slice_3StridedSlice3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_75/strided_slice_3/stack:output:0(lstm_75/strided_slice_3/stack_1:output:0(lstm_75/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskm
lstm_75/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѓ
lstm_75/transpose_1	Transpose3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_75/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€c
lstm_75/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    s
dropout_40/IdentityIdentity lstm_75/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype0С
dense_57/MatMulMatMuldropout_40/Identity:output:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_57/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€£
NoOpNoOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp,^lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp+^lstm_75/lstm_cell_76/MatMul/ReadVariableOp-^lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp^lstm_75/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2Z
+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp2X
*lstm_75/lstm_cell_76/MatMul/ReadVariableOp*lstm_75/lstm_cell_76/MatMul/ReadVariableOp2\
,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp2
lstm_75/whilelstm_75/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
Ё
K__inference_sequential_59_layer_call_and_return_conditional_losses_22141759

inputs"
lstm_75_22141728:"
lstm_75_22141730:
lstm_75_22141732:#
dense_57_22141753:
dense_57_22141755:
identityИҐ dense_57/StatefulPartitionedCallҐlstm_75/StatefulPartitionedCallЖ
lstm_75/StatefulPartitionedCallStatefulPartitionedCallinputslstm_75_22141728lstm_75_22141730lstm_75_22141732*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141727а
dropout_40/PartitionedCallPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141740У
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#dropout_40/PartitionedCall:output:0dense_57_22141753dense_57_22141755*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_57_layer_call_and_return_conditional_losses_22141752x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Л
NoOpNoOp!^dense_57/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
†

н
lstm_75_while_cond_22142176,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3.
*lstm_75_while_less_lstm_75_strided_slice_1F
Blstm_75_while_lstm_75_while_cond_22142176___redundant_placeholder0F
Blstm_75_while_lstm_75_while_cond_22142176___redundant_placeholder1F
Blstm_75_while_lstm_75_while_cond_22142176___redundant_placeholder2F
Blstm_75_while_lstm_75_while_cond_22142176___redundant_placeholder3
lstm_75_while_identity
В
lstm_75/while/LessLesslstm_75_while_placeholder*lstm_75_while_less_lstm_75_strided_slice_1*
T0*
_output_shapes
: [
lstm_75/while/IdentityIdentitylstm_75/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_75_while_identitylstm_75/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
ф
В
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142006

inputs"
lstm_75_22141992:"
lstm_75_22141994:
lstm_75_22141996:#
dense_57_22142000:
dense_57_22142002:
identityИҐ dense_57/StatefulPartitionedCallҐ"dropout_40/StatefulPartitionedCallҐlstm_75/StatefulPartitionedCallЖ
lstm_75/StatefulPartitionedCallStatefulPartitionedCallinputslstm_75_22141992lstm_75_22141994lstm_75_22141996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141963р
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141802Ы
 dense_57/StatefulPartitionedCallStatefulPartitionedCall+dropout_40/StatefulPartitionedCall:output:0dense_57_22142000dense_57_22142002*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_57_layer_call_and_return_conditional_losses_22141752x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€∞
NoOpNoOp!^dense_57/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
±e
О
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142428

inputsE
3lstm_75_lstm_cell_76_matmul_readvariableop_resource:G
5lstm_75_lstm_cell_76_matmul_1_readvariableop_resource:B
4lstm_75_lstm_cell_76_biasadd_readvariableop_resource:9
'dense_57_matmul_readvariableop_resource:6
(dense_57_biasadd_readvariableop_resource:
identityИҐdense_57/BiasAdd/ReadVariableOpҐdense_57/MatMul/ReadVariableOpҐ+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOpҐ*lstm_75/lstm_cell_76/MatMul/ReadVariableOpҐ,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOpҐlstm_75/whileC
lstm_75/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_75/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_75/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_75/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_75/strided_sliceStridedSlicelstm_75/Shape:output:0$lstm_75/strided_slice/stack:output:0&lstm_75/strided_slice/stack_1:output:0&lstm_75/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_75/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Л
lstm_75/zeros/packedPacklstm_75/strided_slice:output:0lstm_75/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_75/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Д
lstm_75/zerosFilllstm_75/zeros/packed:output:0lstm_75/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
lstm_75/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :П
lstm_75/zeros_1/packedPacklstm_75/strided_slice:output:0!lstm_75/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_75/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    К
lstm_75/zeros_1Filllstm_75/zeros_1/packed:output:0lstm_75/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
lstm_75/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_75/transpose	Transposeinputslstm_75/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€T
lstm_75/Shape_1Shapelstm_75/transpose:y:0*
T0*
_output_shapes
:g
lstm_75/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Г
lstm_75/strided_slice_1StridedSlicelstm_75/Shape_1:output:0&lstm_75/strided_slice_1/stack:output:0(lstm_75/strided_slice_1/stack_1:output:0(lstm_75/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_75/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ћ
lstm_75/TensorArrayV2TensorListReserve,lstm_75/TensorArrayV2/element_shape:output:0 lstm_75/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“О
=lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ш
/lstm_75/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_75/transpose:y:0Flstm_75/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“g
lstm_75/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_75/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:С
lstm_75/strided_slice_2StridedSlicelstm_75/transpose:y:0&lstm_75/strided_slice_2/stack:output:0(lstm_75/strided_slice_2/stack_1:output:0(lstm_75/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЮ
*lstm_75/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3lstm_75_lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0≠
lstm_75/lstm_cell_76/MatMulMatMul lstm_75/strided_slice_2:output:02lstm_75/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5lstm_75_lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0І
lstm_75/lstm_cell_76/MatMul_1MatMullstm_75/zeros:output:04lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€£
lstm_75/lstm_cell_76/addAddV2%lstm_75/lstm_cell_76/MatMul:product:0'lstm_75/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4lstm_75_lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
lstm_75/lstm_cell_76/BiasAddBiasAddlstm_75/lstm_cell_76/add:z:03lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
$lstm_75/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :х
lstm_75/lstm_cell_76/splitSplit-lstm_75/lstm_cell_76/split/split_dim:output:0%lstm_75/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_split~
lstm_75/lstm_cell_76/SigmoidSigmoid#lstm_75/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_75/lstm_cell_76/Sigmoid_1Sigmoid#lstm_75/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€П
lstm_75/lstm_cell_76/mulMul"lstm_75/lstm_cell_76/Sigmoid_1:y:0lstm_75/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
lstm_75/lstm_cell_76/ReluRelu#lstm_75/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm_75/lstm_cell_76/mul_1Mul lstm_75/lstm_cell_76/Sigmoid:y:0'lstm_75/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€У
lstm_75/lstm_cell_76/add_1AddV2lstm_75/lstm_cell_76/mul:z:0lstm_75/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€А
lstm_75/lstm_cell_76/Sigmoid_2Sigmoid#lstm_75/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€u
lstm_75/lstm_cell_76/Relu_1Relulstm_75/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
lstm_75/lstm_cell_76/mul_2Mul"lstm_75/lstm_cell_76/Sigmoid_2:y:0)lstm_75/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€v
%lstm_75/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   f
$lstm_75/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_75/TensorArrayV2_1TensorListReserve.lstm_75/TensorArrayV2_1/element_shape:output:0-lstm_75/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“N
lstm_75/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_75/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€\
lstm_75/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ц
lstm_75/whileWhile#lstm_75/while/loop_counter:output:0)lstm_75/while/maximum_iterations:output:0lstm_75/time:output:0 lstm_75/TensorArrayV2_1:handle:0lstm_75/zeros:output:0lstm_75/zeros_1:output:0 lstm_75/strided_slice_1:output:0?lstm_75/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_75_lstm_cell_76_matmul_readvariableop_resource5lstm_75_lstm_cell_76_matmul_1_readvariableop_resource4lstm_75_lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_75_while_body_22142329*'
condR
lstm_75_while_cond_22142328*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Й
8lstm_75/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   о
*lstm_75/TensorArrayV2Stack/TensorListStackTensorListStacklstm_75/while:output:3Alstm_75/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsp
lstm_75/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€i
lstm_75/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_75/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
lstm_75/strided_slice_3StridedSlice3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_75/strided_slice_3/stack:output:0(lstm_75/strided_slice_3/stack_1:output:0(lstm_75/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskm
lstm_75/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѓ
lstm_75/transpose_1	Transpose3lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_75/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€c
lstm_75/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_40/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ф
dropout_40/dropout/MulMul lstm_75/strided_slice_3:output:0!dropout_40/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
dropout_40/dropout/ShapeShape lstm_75/strided_slice_3:output:0*
T0*
_output_shapes
:Ґ
/dropout_40/dropout/random_uniform/RandomUniformRandomUniform!dropout_40/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0f
!dropout_40/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>«
dropout_40/dropout/GreaterEqualGreaterEqual8dropout_40/dropout/random_uniform/RandomUniform:output:0*dropout_40/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
dropout_40/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    њ
dropout_40/dropout/SelectV2SelectV2#dropout_40/dropout/GreaterEqual:z:0dropout_40/dropout/Mul:z:0#dropout_40/dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Щ
dense_57/MatMulMatMul$dropout_40/dropout/SelectV2:output:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_57/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€£
NoOpNoOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp,^lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp+^lstm_75/lstm_cell_76/MatMul/ReadVariableOp-^lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp^lstm_75/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2Z
+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp+lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp2X
*lstm_75/lstm_cell_76/MatMul/ReadVariableOp*lstm_75/lstm_cell_76/MatMul/ReadVariableOp2\
,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp,lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp2
lstm_75/whilelstm_75/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
«\
‘
$__inference__traced_restore_22143355
file_prefix2
 assignvariableop_dense_57_kernel:.
 assignvariableop_1_dense_57_bias:@
.assignvariableop_2_lstm_75_lstm_cell_76_kernel:J
8assignvariableop_3_lstm_75_lstm_cell_76_recurrent_kernel::
,assignvariableop_4_lstm_75_lstm_cell_76_bias:&
assignvariableop_5_iteration:	 *
 assignvariableop_6_learning_rate: G
5assignvariableop_7_adam_m_lstm_75_lstm_cell_76_kernel:G
5assignvariableop_8_adam_v_lstm_75_lstm_cell_76_kernel:Q
?assignvariableop_9_adam_m_lstm_75_lstm_cell_76_recurrent_kernel:R
@assignvariableop_10_adam_v_lstm_75_lstm_cell_76_recurrent_kernel:B
4assignvariableop_11_adam_m_lstm_75_lstm_cell_76_bias:B
4assignvariableop_12_adam_v_lstm_75_lstm_cell_76_bias:<
*assignvariableop_13_adam_m_dense_57_kernel:<
*assignvariableop_14_adam_v_dense_57_kernel:6
(assignvariableop_15_adam_m_dense_57_bias:6
(assignvariableop_16_adam_v_dense_57_bias:%
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: 
identity_22ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9±	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*„
valueЌB B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B М
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOpAssignVariableOp assignvariableop_dense_57_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_57_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_75_lstm_cell_76_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_75_lstm_cell_76_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_75_lstm_cell_76_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:≥
AssignVariableOp_5AssignVariableOpassignvariableop_5_iterationIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_6AssignVariableOp assignvariableop_6_learning_rateIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_7AssignVariableOp5assignvariableop_7_adam_m_lstm_75_lstm_cell_76_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_8AssignVariableOp5assignvariableop_8_adam_v_lstm_75_lstm_cell_76_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:÷
AssignVariableOp_9AssignVariableOp?assignvariableop_9_adam_m_lstm_75_lstm_cell_76_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_10AssignVariableOp@assignvariableop_10_adam_v_lstm_75_lstm_cell_76_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_m_lstm_75_lstm_cell_76_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_v_lstm_75_lstm_cell_76_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_m_dense_57_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_v_dense_57_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_m_dense_57_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_v_dense_57_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Э
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: К
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
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
Ю$
л
while_body_22141497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_76_22141521_0:/
while_lstm_cell_76_22141523_0:+
while_lstm_cell_76_22141525_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_76_22141521:-
while_lstm_cell_76_22141523:)
while_lstm_cell_76_22141525:ИҐ*while/lstm_cell_76/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ї
*while/lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_76_22141521_0while_lstm_cell_76_22141523_0while_lstm_cell_76_22141525_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141437r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Д
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_76/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Р
while/Identity_4Identity3while/lstm_cell_76/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Р
while/Identity_5Identity3while/lstm_cell_76/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y

while/NoOpNoOp+^while/lstm_cell_76/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_76_22141521while_lstm_cell_76_22141521_0"<
while_lstm_cell_76_22141523while_lstm_cell_76_22141523_0"<
while_lstm_cell_76_22141525while_lstm_cell_76_22141525_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2X
*while/lstm_cell_76/StatefulPartitionedCall*while/lstm_cell_76/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
ёB
ћ

lstm_75_while_body_22142177,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3+
'lstm_75_while_lstm_75_strided_slice_1_0g
clstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0:O
=lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0:J
<lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0:
lstm_75_while_identity
lstm_75_while_identity_1
lstm_75_while_identity_2
lstm_75_while_identity_3
lstm_75_while_identity_4
lstm_75_while_identity_5)
%lstm_75_while_lstm_75_strided_slice_1e
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorK
9lstm_75_while_lstm_cell_76_matmul_readvariableop_resource:M
;lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource:H
:lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOpҐ0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOpҐ2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOpР
?lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ќ
1lstm_75/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0lstm_75_while_placeholderHlstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ђ
0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp;lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0—
!lstm_75/while/lstm_cell_76/MatMulMatMul8lstm_75/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€∞
2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp=lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Є
#lstm_75/while/lstm_cell_76/MatMul_1MatMullstm_75_while_placeholder_2:lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€µ
lstm_75/while/lstm_cell_76/addAddV2+lstm_75/while/lstm_cell_76/MatMul:product:0-lstm_75/while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€™
1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp<lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
"lstm_75/while/lstm_cell_76/BiasAddBiasAdd"lstm_75/while/lstm_cell_76/add:z:09lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€l
*lstm_75/while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :З
 lstm_75/while/lstm_cell_76/splitSplit3lstm_75/while/lstm_cell_76/split/split_dim:output:0+lstm_75/while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitК
"lstm_75/while/lstm_cell_76/SigmoidSigmoid)lstm_75/while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€М
$lstm_75/while/lstm_cell_76/Sigmoid_1Sigmoid)lstm_75/while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm_75/while/lstm_cell_76/mulMul(lstm_75/while/lstm_cell_76/Sigmoid_1:y:0lstm_75_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€Д
lstm_75/while/lstm_cell_76/ReluRelu)lstm_75/while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€∞
 lstm_75/while/lstm_cell_76/mul_1Mul&lstm_75/while/lstm_cell_76/Sigmoid:y:0-lstm_75/while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€•
 lstm_75/while/lstm_cell_76/add_1AddV2"lstm_75/while/lstm_cell_76/mul:z:0$lstm_75/while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€М
$lstm_75/while/lstm_cell_76/Sigmoid_2Sigmoid)lstm_75/while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€Б
!lstm_75/while/lstm_cell_76/Relu_1Relu$lstm_75/while/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€і
 lstm_75/while/lstm_cell_76/mul_2Mul(lstm_75/while/lstm_cell_76/Sigmoid_2:y:0/lstm_75/while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€z
8lstm_75/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Н
2lstm_75/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_75_while_placeholder_1Alstm_75/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_75/while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“U
lstm_75/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_75/while/addAddV2lstm_75_while_placeholderlstm_75/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_75/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_75/while/add_1AddV2(lstm_75_while_lstm_75_while_loop_counterlstm_75/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_75/while/IdentityIdentitylstm_75/while/add_1:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: К
lstm_75/while/Identity_1Identity.lstm_75_while_lstm_75_while_maximum_iterations^lstm_75/while/NoOp*
T0*
_output_shapes
: q
lstm_75/while/Identity_2Identitylstm_75/while/add:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: Ю
lstm_75/while/Identity_3IdentityBlstm_75/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_75/while/NoOp*
T0*
_output_shapes
: С
lstm_75/while/Identity_4Identity$lstm_75/while/lstm_cell_76/mul_2:z:0^lstm_75/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
lstm_75/while/Identity_5Identity$lstm_75/while/lstm_cell_76/add_1:z:0^lstm_75/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€р
lstm_75/while/NoOpNoOp2^lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp1^lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp3^lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_75_while_identitylstm_75/while/Identity:output:0"=
lstm_75_while_identity_1!lstm_75/while/Identity_1:output:0"=
lstm_75_while_identity_2!lstm_75/while/Identity_2:output:0"=
lstm_75_while_identity_3!lstm_75/while/Identity_3:output:0"=
lstm_75_while_identity_4!lstm_75/while/Identity_4:output:0"=
lstm_75_while_identity_5!lstm_75/while/Identity_5:output:0"P
%lstm_75_while_lstm_75_strided_slice_1'lstm_75_while_lstm_75_strided_slice_1_0"z
:lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource<lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0"|
;lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource=lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0"x
9lstm_75_while_lstm_cell_76_matmul_readvariableop_resource;lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0"»
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2f
1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp2d
0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp2h
2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ф9
К
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141567

inputs'
lstm_cell_76_22141483:'
lstm_cell_76_22141485:#
lstm_cell_76_22141487:
identityИҐ$lstm_cell_76/StatefulPartitionedCallҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskэ
$lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_76_22141483lstm_cell_76_22141485lstm_cell_76_22141487*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141437n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_76_22141483lstm_cell_76_22141485lstm_cell_76_22141487*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22141497*
condR
while_cond_22141496*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€u
NoOpNoOp%^lstm_cell_76/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2L
$lstm_cell_76/StatefulPartitionedCall$lstm_cell_76/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
»K
Ь
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142762
inputs_0=
+lstm_cell_76_matmul_readvariableop_resource:?
-lstm_cell_76_matmul_1_readvariableop_resource::
,lstm_cell_76_biasadd_readvariableop_resource:
identityИҐ#lstm_cell_76/BiasAdd/ReadVariableOpҐ"lstm_cell_76/MatMul/ReadVariableOpҐ$lstm_cell_76/MatMul_1/ReadVariableOpҐwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskО
"lstm_cell_76/MatMul/ReadVariableOpReadVariableOp+lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Х
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0*lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
$lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_cell_76/MatMul_1MatMulzeros:output:0,lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Л
lstm_cell_76/addAddV2lstm_cell_76/MatMul:product:0lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€М
#lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
lstm_cell_76/BiasAddBiasAddlstm_cell_76/add:z:0+lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitn
lstm_cell_76/SigmoidSigmoidlstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€w
lstm_cell_76/mulMullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm_cell_76/ReluRelulstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell_76/mul_1Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm_cell_76/add_1AddV2lstm_cell_76/mul:z:0lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell_76/Relu_1Relulstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell_76/mul_2Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_76_matmul_readvariableop_resource-lstm_cell_76_matmul_1_readvariableop_resource,lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22142677*
condR
while_cond_22142676*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ј
NoOpNoOp$^lstm_cell_76/BiasAdd/ReadVariableOp#^lstm_cell_76/MatMul/ReadVariableOp%^lstm_cell_76/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2J
#lstm_cell_76/BiasAdd/ReadVariableOp#lstm_cell_76/BiasAdd/ReadVariableOp2H
"lstm_cell_76/MatMul/ReadVariableOp"lstm_cell_76/MatMul/ReadVariableOp2L
$lstm_cell_76/MatMul_1/ReadVariableOp$lstm_cell_76/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
’
Е
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143164

inputs
states_0
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€N
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
њ
Ќ
while_cond_22141303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22141303___redundant_placeholder06
2while_while_cond_22141303___redundant_placeholder16
2while_while_cond_22141303___redundant_placeholder26
2while_while_cond_22141303___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
м
х
/__inference_lstm_cell_76_layer_call_fn_22143132

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2ИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141437o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
†

н
lstm_75_while_cond_22142328,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3.
*lstm_75_while_less_lstm_75_strided_slice_1F
Blstm_75_while_lstm_75_while_cond_22142328___redundant_placeholder0F
Blstm_75_while_lstm_75_while_cond_22142328___redundant_placeholder1F
Blstm_75_while_lstm_75_while_cond_22142328___redundant_placeholder2F
Blstm_75_while_lstm_75_while_cond_22142328___redundant_placeholder3
lstm_75_while_identity
В
lstm_75/while/LessLesslstm_75_while_placeholder*lstm_75_while_less_lstm_75_strided_slice_1*
T0*
_output_shapes
: [
lstm_75/while/IdentityIdentitylstm_75/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_75_while_identitylstm_75/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Й
Й
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142068
lstm_75_input"
lstm_75_22142054:"
lstm_75_22142056:
lstm_75_22142058:#
dense_57_22142062:
dense_57_22142064:
identityИҐ dense_57/StatefulPartitionedCallҐ"dropout_40/StatefulPartitionedCallҐlstm_75/StatefulPartitionedCallН
lstm_75/StatefulPartitionedCallStatefulPartitionedCalllstm_75_inputlstm_75_22142054lstm_75_22142056lstm_75_22142058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141963р
"dropout_40/StatefulPartitionedCallStatefulPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141802Ы
 dense_57/StatefulPartitionedCallStatefulPartitionedCall+dropout_40/StatefulPartitionedCall:output:0dense_57_22142062dense_57_22142064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_57_layer_call_and_return_conditional_losses_22141752x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€∞
NoOpNoOp!^dense_57/StatefulPartitionedCall#^dropout_40/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2H
"dropout_40/StatefulPartitionedCall"dropout_40/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€
'
_user_specified_namelstm_75_input
њ
Ќ
while_cond_22141496
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22141496___redundant_placeholder06
2while_while_cond_22141496___redundant_placeholder16
2while_while_cond_22141496___redundant_placeholder26
2while_while_cond_22141496___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
∆
Ш
+__inference_dense_57_layer_call_fn_22143088

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_57_layer_call_and_return_conditional_losses_22141752o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э9
ћ
while_body_22141642
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_76_matmul_readvariableop_resource_0:G
5while_lstm_cell_76_matmul_1_readvariableop_resource_0:B
4while_lstm_cell_76_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_76_matmul_readvariableop_resource:E
3while_lstm_cell_76_matmul_1_readvariableop_resource:@
2while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ)while/lstm_cell_76/BiasAdd/ReadVariableOpҐ(while/lstm_cell_76/MatMul/ReadVariableOpҐ*while/lstm_cell_76/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ь
(while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0є
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
*while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0†
while/lstm_cell_76/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
while/lstm_cell_76/addAddV2#while/lstm_cell_76/MatMul:product:0%while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
)while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¶
while/lstm_cell_76/BiasAddBiasAddwhile/lstm_cell_76/add:z:01while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0#while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitz
while/lstm_cell_76/SigmoidSigmoid!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_1Sigmoid!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell_76/mulMul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€t
while/lstm_cell_76/ReluRelu!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell_76/mul_1Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Н
while/lstm_cell_76/add_1AddV2while/lstm_cell_76/mul:z:0while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_2Sigmoid!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell_76/mul_2Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : н
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_76/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y
while/Identity_5Identitywhile/lstm_cell_76/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€–

while/NoOpNoOp*^while/lstm_cell_76/BiasAdd/ReadVariableOp)^while/lstm_cell_76/MatMul/ReadVariableOp+^while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_76_biasadd_readvariableop_resource4while_lstm_cell_76_biasadd_readvariableop_resource_0"l
3while_lstm_cell_76_matmul_1_readvariableop_resource5while_lstm_cell_76_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_76_matmul_readvariableop_resource3while_lstm_cell_76_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2V
)while/lstm_cell_76/BiasAdd/ReadVariableOp)while/lstm_cell_76/BiasAdd/ReadVariableOp2T
(while/lstm_cell_76/MatMul/ReadVariableOp(while/lstm_cell_76/MatMul/ReadVariableOp2X
*while/lstm_cell_76/MatMul_1/ReadVariableOp*while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
…	
ч
F__inference_dense_57_layer_call_and_return_conditional_losses_22143098

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м
х
/__inference_lstm_cell_76_layer_call_fn_22143115

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2ИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141289o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
Р
ґ
*__inference_lstm_75_layer_call_fn_22142450
inputs_0
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
®
Е
)sequential_59_lstm_75_while_cond_22141129H
Dsequential_59_lstm_75_while_sequential_59_lstm_75_while_loop_counterN
Jsequential_59_lstm_75_while_sequential_59_lstm_75_while_maximum_iterations+
'sequential_59_lstm_75_while_placeholder-
)sequential_59_lstm_75_while_placeholder_1-
)sequential_59_lstm_75_while_placeholder_2-
)sequential_59_lstm_75_while_placeholder_3J
Fsequential_59_lstm_75_while_less_sequential_59_lstm_75_strided_slice_1b
^sequential_59_lstm_75_while_sequential_59_lstm_75_while_cond_22141129___redundant_placeholder0b
^sequential_59_lstm_75_while_sequential_59_lstm_75_while_cond_22141129___redundant_placeholder1b
^sequential_59_lstm_75_while_sequential_59_lstm_75_while_cond_22141129___redundant_placeholder2b
^sequential_59_lstm_75_while_sequential_59_lstm_75_while_cond_22141129___redundant_placeholder3(
$sequential_59_lstm_75_while_identity
Ї
 sequential_59/lstm_75/while/LessLess'sequential_59_lstm_75_while_placeholderFsequential_59_lstm_75_while_less_sequential_59_lstm_75_strided_slice_1*
T0*
_output_shapes
: w
$sequential_59/lstm_75/while/IdentityIdentity$sequential_59/lstm_75/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_59_lstm_75_while_identity-sequential_59/lstm_75/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ф9
К
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141374

inputs'
lstm_cell_76_22141290:'
lstm_cell_76_22141292:#
lstm_cell_76_22141294:
identityИҐ$lstm_cell_76/StatefulPartitionedCallҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskэ
$lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_76_22141290lstm_cell_76_22141292lstm_cell_76_22141294*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141289n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_76_22141290lstm_cell_76_22141292lstm_cell_76_22141294*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22141304*
condR
while_cond_22141303*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€u
NoOpNoOp%^lstm_cell_76/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2L
$lstm_cell_76/StatefulPartitionedCall$lstm_cell_76/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
•K
Ъ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141963

inputs=
+lstm_cell_76_matmul_readvariableop_resource:?
-lstm_cell_76_matmul_1_readvariableop_resource::
,lstm_cell_76_biasadd_readvariableop_resource:
identityИҐ#lstm_cell_76/BiasAdd/ReadVariableOpҐ"lstm_cell_76/MatMul/ReadVariableOpҐ$lstm_cell_76/MatMul_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskО
"lstm_cell_76/MatMul/ReadVariableOpReadVariableOp+lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Х
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0*lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
$lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_cell_76/MatMul_1MatMulzeros:output:0,lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Л
lstm_cell_76/addAddV2lstm_cell_76/MatMul:product:0lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€М
#lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
lstm_cell_76/BiasAddBiasAddlstm_cell_76/add:z:0+lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitn
lstm_cell_76/SigmoidSigmoidlstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€w
lstm_cell_76/mulMullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm_cell_76/ReluRelulstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell_76/mul_1Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm_cell_76/add_1AddV2lstm_cell_76/mul:z:0lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell_76/Relu_1Relulstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell_76/mul_2Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_76_matmul_readvariableop_resource-lstm_cell_76_matmul_1_readvariableop_resource,lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22141878*
condR
while_cond_22141877*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ј
NoOpNoOp$^lstm_cell_76/BiasAdd/ReadVariableOp#^lstm_cell_76/MatMul/ReadVariableOp%^lstm_cell_76/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2J
#lstm_cell_76/BiasAdd/ReadVariableOp#lstm_cell_76/BiasAdd/ReadVariableOp2H
"lstm_cell_76/MatMul/ReadVariableOp"lstm_cell_76/MatMul/ReadVariableOp2L
$lstm_cell_76/MatMul_1/ReadVariableOp$lstm_cell_76/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
н
ч
0__inference_sequential_59_layer_call_fn_22142034
lstm_75_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCalllstm_75_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142006o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€
'
_user_specified_namelstm_75_input
Э9
ћ
while_body_22142822
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_76_matmul_readvariableop_resource_0:G
5while_lstm_cell_76_matmul_1_readvariableop_resource_0:B
4while_lstm_cell_76_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_76_matmul_readvariableop_resource:E
3while_lstm_cell_76_matmul_1_readvariableop_resource:@
2while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ)while/lstm_cell_76/BiasAdd/ReadVariableOpҐ(while/lstm_cell_76/MatMul/ReadVariableOpҐ*while/lstm_cell_76/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ь
(while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0є
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
*while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0†
while/lstm_cell_76/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
while/lstm_cell_76/addAddV2#while/lstm_cell_76/MatMul:product:0%while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
)while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¶
while/lstm_cell_76/BiasAddBiasAddwhile/lstm_cell_76/add:z:01while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0#while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitz
while/lstm_cell_76/SigmoidSigmoid!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_1Sigmoid!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell_76/mulMul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€t
while/lstm_cell_76/ReluRelu!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell_76/mul_1Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Н
while/lstm_cell_76/add_1AddV2while/lstm_cell_76/mul:z:0while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_2Sigmoid!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell_76/mul_2Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : н
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_76/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y
while/Identity_5Identitywhile/lstm_cell_76/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€–

while/NoOpNoOp*^while/lstm_cell_76/BiasAdd/ReadVariableOp)^while/lstm_cell_76/MatMul/ReadVariableOp+^while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_76_biasadd_readvariableop_resource4while_lstm_cell_76_biasadd_readvariableop_resource_0"l
3while_lstm_cell_76_matmul_1_readvariableop_resource5while_lstm_cell_76_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_76_matmul_readvariableop_resource3while_lstm_cell_76_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2V
)while/lstm_cell_76/BiasAdd/ReadVariableOp)while/lstm_cell_76/BiasAdd/ReadVariableOp2T
(while/lstm_cell_76/MatMul/ReadVariableOp(while/lstm_cell_76/MatMul/ReadVariableOp2X
*while/lstm_cell_76/MatMul_1/ReadVariableOp*while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Э9
ћ
while_body_22142677
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_76_matmul_readvariableop_resource_0:G
5while_lstm_cell_76_matmul_1_readvariableop_resource_0:B
4while_lstm_cell_76_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_76_matmul_readvariableop_resource:E
3while_lstm_cell_76_matmul_1_readvariableop_resource:@
2while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ)while/lstm_cell_76/BiasAdd/ReadVariableOpҐ(while/lstm_cell_76/MatMul/ReadVariableOpҐ*while/lstm_cell_76/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ь
(while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0є
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
*while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0†
while/lstm_cell_76/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
while/lstm_cell_76/addAddV2#while/lstm_cell_76/MatMul:product:0%while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
)while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¶
while/lstm_cell_76/BiasAddBiasAddwhile/lstm_cell_76/add:z:01while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0#while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitz
while/lstm_cell_76/SigmoidSigmoid!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_1Sigmoid!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell_76/mulMul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€t
while/lstm_cell_76/ReluRelu!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell_76/mul_1Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Н
while/lstm_cell_76/add_1AddV2while/lstm_cell_76/mul:z:0while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_2Sigmoid!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell_76/mul_2Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : н
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_76/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y
while/Identity_5Identitywhile/lstm_cell_76/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€–

while/NoOpNoOp*^while/lstm_cell_76/BiasAdd/ReadVariableOp)^while/lstm_cell_76/MatMul/ReadVariableOp+^while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_76_biasadd_readvariableop_resource4while_lstm_cell_76_biasadd_readvariableop_resource_0"l
3while_lstm_cell_76_matmul_1_readvariableop_resource5while_lstm_cell_76_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_76_matmul_readvariableop_resource3while_lstm_cell_76_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2V
)while/lstm_cell_76/BiasAdd/ReadVariableOp)while/lstm_cell_76/BiasAdd/ReadVariableOp2T
(while/lstm_cell_76/MatMul/ReadVariableOp(while/lstm_cell_76/MatMul/ReadVariableOp2X
*while/lstm_cell_76/MatMul_1/ReadVariableOp*while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
≠S
О
)sequential_59_lstm_75_while_body_22141130H
Dsequential_59_lstm_75_while_sequential_59_lstm_75_while_loop_counterN
Jsequential_59_lstm_75_while_sequential_59_lstm_75_while_maximum_iterations+
'sequential_59_lstm_75_while_placeholder-
)sequential_59_lstm_75_while_placeholder_1-
)sequential_59_lstm_75_while_placeholder_2-
)sequential_59_lstm_75_while_placeholder_3G
Csequential_59_lstm_75_while_sequential_59_lstm_75_strided_slice_1_0Г
sequential_59_lstm_75_while_tensorarrayv2read_tensorlistgetitem_sequential_59_lstm_75_tensorarrayunstack_tensorlistfromtensor_0[
Isequential_59_lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0:]
Ksequential_59_lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0:X
Jsequential_59_lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0:(
$sequential_59_lstm_75_while_identity*
&sequential_59_lstm_75_while_identity_1*
&sequential_59_lstm_75_while_identity_2*
&sequential_59_lstm_75_while_identity_3*
&sequential_59_lstm_75_while_identity_4*
&sequential_59_lstm_75_while_identity_5E
Asequential_59_lstm_75_while_sequential_59_lstm_75_strided_slice_1Б
}sequential_59_lstm_75_while_tensorarrayv2read_tensorlistgetitem_sequential_59_lstm_75_tensorarrayunstack_tensorlistfromtensorY
Gsequential_59_lstm_75_while_lstm_cell_76_matmul_readvariableop_resource:[
Isequential_59_lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource:V
Hsequential_59_lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ?sequential_59/lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOpҐ>sequential_59/lstm_75/while/lstm_cell_76/MatMul/ReadVariableOpҐ@sequential_59/lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOpЮ
Msequential_59/lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ф
?sequential_59/lstm_75/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_59_lstm_75_while_tensorarrayv2read_tensorlistgetitem_sequential_59_lstm_75_tensorarrayunstack_tensorlistfromtensor_0'sequential_59_lstm_75_while_placeholderVsequential_59/lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0»
>sequential_59/lstm_75/while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOpIsequential_59_lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0ы
/sequential_59/lstm_75/while/lstm_cell_76/MatMulMatMulFsequential_59/lstm_75/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_59/lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ћ
@sequential_59/lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOpKsequential_59_lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0в
1sequential_59/lstm_75/while/lstm_cell_76/MatMul_1MatMul)sequential_59_lstm_75_while_placeholder_2Hsequential_59/lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€я
,sequential_59/lstm_75/while/lstm_cell_76/addAddV29sequential_59/lstm_75/while/lstm_cell_76/MatMul:product:0;sequential_59/lstm_75/while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€∆
?sequential_59/lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOpJsequential_59_lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0и
0sequential_59/lstm_75/while/lstm_cell_76/BiasAddBiasAdd0sequential_59/lstm_75/while/lstm_cell_76/add:z:0Gsequential_59/lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€z
8sequential_59/lstm_75/while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.sequential_59/lstm_75/while/lstm_cell_76/splitSplitAsequential_59/lstm_75/while/lstm_cell_76/split/split_dim:output:09sequential_59/lstm_75/while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_split¶
0sequential_59/lstm_75/while/lstm_cell_76/SigmoidSigmoid7sequential_59/lstm_75/while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€®
2sequential_59/lstm_75/while/lstm_cell_76/Sigmoid_1Sigmoid7sequential_59/lstm_75/while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€»
,sequential_59/lstm_75/while/lstm_cell_76/mulMul6sequential_59/lstm_75/while/lstm_cell_76/Sigmoid_1:y:0)sequential_59_lstm_75_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€†
-sequential_59/lstm_75/while/lstm_cell_76/ReluRelu7sequential_59/lstm_75/while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Џ
.sequential_59/lstm_75/while/lstm_cell_76/mul_1Mul4sequential_59/lstm_75/while/lstm_cell_76/Sigmoid:y:0;sequential_59/lstm_75/while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ѕ
.sequential_59/lstm_75/while/lstm_cell_76/add_1AddV20sequential_59/lstm_75/while/lstm_cell_76/mul:z:02sequential_59/lstm_75/while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€®
2sequential_59/lstm_75/while/lstm_cell_76/Sigmoid_2Sigmoid7sequential_59/lstm_75/while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€Э
/sequential_59/lstm_75/while/lstm_cell_76/Relu_1Relu2sequential_59/lstm_75/while/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ё
.sequential_59/lstm_75/while/lstm_cell_76/mul_2Mul6sequential_59/lstm_75/while/lstm_cell_76/Sigmoid_2:y:0=sequential_59/lstm_75/while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€И
Fsequential_59/lstm_75/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ≈
@sequential_59/lstm_75/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_59_lstm_75_while_placeholder_1Osequential_59/lstm_75/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_59/lstm_75/while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“c
!sequential_59/lstm_75/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ю
sequential_59/lstm_75/while/addAddV2'sequential_59_lstm_75_while_placeholder*sequential_59/lstm_75/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_59/lstm_75/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :њ
!sequential_59/lstm_75/while/add_1AddV2Dsequential_59_lstm_75_while_sequential_59_lstm_75_while_loop_counter,sequential_59/lstm_75/while/add_1/y:output:0*
T0*
_output_shapes
: Ы
$sequential_59/lstm_75/while/IdentityIdentity%sequential_59/lstm_75/while/add_1:z:0!^sequential_59/lstm_75/while/NoOp*
T0*
_output_shapes
: ¬
&sequential_59/lstm_75/while/Identity_1IdentityJsequential_59_lstm_75_while_sequential_59_lstm_75_while_maximum_iterations!^sequential_59/lstm_75/while/NoOp*
T0*
_output_shapes
: Ы
&sequential_59/lstm_75/while/Identity_2Identity#sequential_59/lstm_75/while/add:z:0!^sequential_59/lstm_75/while/NoOp*
T0*
_output_shapes
: »
&sequential_59/lstm_75/while/Identity_3IdentityPsequential_59/lstm_75/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_59/lstm_75/while/NoOp*
T0*
_output_shapes
: ї
&sequential_59/lstm_75/while/Identity_4Identity2sequential_59/lstm_75/while/lstm_cell_76/mul_2:z:0!^sequential_59/lstm_75/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ї
&sequential_59/lstm_75/while/Identity_5Identity2sequential_59/lstm_75/while/lstm_cell_76/add_1:z:0!^sequential_59/lstm_75/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€®
 sequential_59/lstm_75/while/NoOpNoOp@^sequential_59/lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp?^sequential_59/lstm_75/while/lstm_cell_76/MatMul/ReadVariableOpA^sequential_59/lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_59_lstm_75_while_identity-sequential_59/lstm_75/while/Identity:output:0"Y
&sequential_59_lstm_75_while_identity_1/sequential_59/lstm_75/while/Identity_1:output:0"Y
&sequential_59_lstm_75_while_identity_2/sequential_59/lstm_75/while/Identity_2:output:0"Y
&sequential_59_lstm_75_while_identity_3/sequential_59/lstm_75/while/Identity_3:output:0"Y
&sequential_59_lstm_75_while_identity_4/sequential_59/lstm_75/while/Identity_4:output:0"Y
&sequential_59_lstm_75_while_identity_5/sequential_59/lstm_75/while/Identity_5:output:0"Ц
Hsequential_59_lstm_75_while_lstm_cell_76_biasadd_readvariableop_resourceJsequential_59_lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0"Ш
Isequential_59_lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resourceKsequential_59_lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0"Ф
Gsequential_59_lstm_75_while_lstm_cell_76_matmul_readvariableop_resourceIsequential_59_lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0"И
Asequential_59_lstm_75_while_sequential_59_lstm_75_strided_slice_1Csequential_59_lstm_75_while_sequential_59_lstm_75_strided_slice_1_0"А
}sequential_59_lstm_75_while_tensorarrayv2read_tensorlistgetitem_sequential_59_lstm_75_tensorarrayunstack_tensorlistfromtensorsequential_59_lstm_75_while_tensorarrayv2read_tensorlistgetitem_sequential_59_lstm_75_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2В
?sequential_59/lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp?sequential_59/lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp2А
>sequential_59/lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp>sequential_59/lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp2Д
@sequential_59/lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp@sequential_59/lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
’
Е
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143196

inputs
states_0
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€N
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
states_1
њ
Ќ
while_cond_22142966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22142966___redundant_placeholder06
2while_while_cond_22142966___redundant_placeholder16
2while_while_cond_22142966___redundant_placeholder26
2while_while_cond_22142966___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Р
ґ
*__inference_lstm_75_layer_call_fn_22142439
inputs_0
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
»K
Ь
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142617
inputs_0=
+lstm_cell_76_matmul_readvariableop_resource:?
-lstm_cell_76_matmul_1_readvariableop_resource::
,lstm_cell_76_biasadd_readvariableop_resource:
identityИҐ#lstm_cell_76/BiasAdd/ReadVariableOpҐ"lstm_cell_76/MatMul/ReadVariableOpҐ$lstm_cell_76/MatMul_1/ReadVariableOpҐwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskО
"lstm_cell_76/MatMul/ReadVariableOpReadVariableOp+lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Х
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0*lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
$lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_cell_76/MatMul_1MatMulzeros:output:0,lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Л
lstm_cell_76/addAddV2lstm_cell_76/MatMul:product:0lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€М
#lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
lstm_cell_76/BiasAddBiasAddlstm_cell_76/add:z:0+lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitn
lstm_cell_76/SigmoidSigmoidlstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€w
lstm_cell_76/mulMullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm_cell_76/ReluRelulstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell_76/mul_1Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm_cell_76/add_1AddV2lstm_cell_76/mul:z:0lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell_76/Relu_1Relulstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell_76/mul_2Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_76_matmul_readvariableop_resource-lstm_cell_76_matmul_1_readvariableop_resource,lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22142532*
condR
while_cond_22142531*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ј
NoOpNoOp$^lstm_cell_76/BiasAdd/ReadVariableOp#^lstm_cell_76/MatMul/ReadVariableOp%^lstm_cell_76/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2J
#lstm_cell_76/BiasAdd/ReadVariableOp#lstm_cell_76/BiasAdd/ReadVariableOp2H
"lstm_cell_76/MatMul/ReadVariableOp"lstm_cell_76/MatMul/ReadVariableOp2L
$lstm_cell_76/MatMul_1/ReadVariableOp$lstm_cell_76/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
Э9
ћ
while_body_22142532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_76_matmul_readvariableop_resource_0:G
5while_lstm_cell_76_matmul_1_readvariableop_resource_0:B
4while_lstm_cell_76_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_76_matmul_readvariableop_resource:E
3while_lstm_cell_76_matmul_1_readvariableop_resource:@
2while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ)while/lstm_cell_76/BiasAdd/ReadVariableOpҐ(while/lstm_cell_76/MatMul/ReadVariableOpҐ*while/lstm_cell_76/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ь
(while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0є
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
*while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0†
while/lstm_cell_76/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
while/lstm_cell_76/addAddV2#while/lstm_cell_76/MatMul:product:0%while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
)while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¶
while/lstm_cell_76/BiasAddBiasAddwhile/lstm_cell_76/add:z:01while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0#while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitz
while/lstm_cell_76/SigmoidSigmoid!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_1Sigmoid!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell_76/mulMul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€t
while/lstm_cell_76/ReluRelu!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell_76/mul_1Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Н
while/lstm_cell_76/add_1AddV2while/lstm_cell_76/mul:z:0while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_2Sigmoid!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell_76/mul_2Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : н
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_76/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y
while/Identity_5Identitywhile/lstm_cell_76/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€–

while/NoOpNoOp*^while/lstm_cell_76/BiasAdd/ReadVariableOp)^while/lstm_cell_76/MatMul/ReadVariableOp+^while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_76_biasadd_readvariableop_resource4while_lstm_cell_76_biasadd_readvariableop_resource_0"l
3while_lstm_cell_76_matmul_1_readvariableop_resource5while_lstm_cell_76_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_76_matmul_readvariableop_resource3while_lstm_cell_76_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2V
)while/lstm_cell_76/BiasAdd/ReadVariableOp)while/lstm_cell_76/BiasAdd/ReadVariableOp2T
(while/lstm_cell_76/MatMul/ReadVariableOp(while/lstm_cell_76/MatMul/ReadVariableOp2X
*while/lstm_cell_76/MatMul_1/ReadVariableOp*while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
Ќ
Г
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141289

inputs

states
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€N
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
ш
і
*__inference_lstm_75_layer_call_fn_22142472

inputs
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141963o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
з2
ц	
!__inference__traced_save_22143282
file_prefix.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop:
6savev2_lstm_75_lstm_cell_76_kernel_read_readvariableopD
@savev2_lstm_75_lstm_cell_76_recurrent_kernel_read_readvariableop8
4savev2_lstm_75_lstm_cell_76_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_lstm_75_lstm_cell_76_kernel_read_readvariableopA
=savev2_adam_v_lstm_75_lstm_cell_76_kernel_read_readvariableopK
Gsavev2_adam_m_lstm_75_lstm_cell_76_recurrent_kernel_read_readvariableopK
Gsavev2_adam_v_lstm_75_lstm_cell_76_recurrent_kernel_read_readvariableop?
;savev2_adam_m_lstm_75_lstm_cell_76_bias_read_readvariableop?
;savev2_adam_v_lstm_75_lstm_cell_76_bias_read_readvariableop5
1savev2_adam_m_dense_57_kernel_read_readvariableop5
1savev2_adam_v_dense_57_kernel_read_readvariableop3
/savev2_adam_m_dense_57_bias_read_readvariableop3
/savev2_adam_v_dense_57_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѓ	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*„
valueЌB B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЩ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B °

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop6savev2_lstm_75_lstm_cell_76_kernel_read_readvariableop@savev2_lstm_75_lstm_cell_76_recurrent_kernel_read_readvariableop4savev2_lstm_75_lstm_cell_76_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_lstm_75_lstm_cell_76_kernel_read_readvariableop=savev2_adam_v_lstm_75_lstm_cell_76_kernel_read_readvariableopGsavev2_adam_m_lstm_75_lstm_cell_76_recurrent_kernel_read_readvariableopGsavev2_adam_v_lstm_75_lstm_cell_76_recurrent_kernel_read_readvariableop;savev2_adam_m_lstm_75_lstm_cell_76_bias_read_readvariableop;savev2_adam_v_lstm_75_lstm_cell_76_bias_read_readvariableop1savev2_adam_m_dense_57_kernel_read_readvariableop1savev2_adam_v_dense_57_kernel_read_readvariableop/savev2_adam_m_dense_57_bias_read_readvariableop/savev2_adam_v_dense_57_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *$
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*£
_input_shapesС
О: :::::: : ::::::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$	 

_output_shapes

::$
 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
•K
Ъ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22143052

inputs=
+lstm_cell_76_matmul_readvariableop_resource:?
-lstm_cell_76_matmul_1_readvariableop_resource::
,lstm_cell_76_biasadd_readvariableop_resource:
identityИҐ#lstm_cell_76/BiasAdd/ReadVariableOpҐ"lstm_cell_76/MatMul/ReadVariableOpҐ$lstm_cell_76/MatMul_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskО
"lstm_cell_76/MatMul/ReadVariableOpReadVariableOp+lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Х
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0*lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
$lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_cell_76/MatMul_1MatMulzeros:output:0,lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Л
lstm_cell_76/addAddV2lstm_cell_76/MatMul:product:0lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€М
#lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
lstm_cell_76/BiasAddBiasAddlstm_cell_76/add:z:0+lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitn
lstm_cell_76/SigmoidSigmoidlstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€w
lstm_cell_76/mulMullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm_cell_76/ReluRelulstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell_76/mul_1Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm_cell_76/add_1AddV2lstm_cell_76/mul:z:0lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell_76/Relu_1Relulstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell_76/mul_2Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_76_matmul_readvariableop_resource-lstm_cell_76_matmul_1_readvariableop_resource,lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22142967*
condR
while_cond_22142966*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ј
NoOpNoOp$^lstm_cell_76/BiasAdd/ReadVariableOp#^lstm_cell_76/MatMul/ReadVariableOp%^lstm_cell_76/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2J
#lstm_cell_76/BiasAdd/ReadVariableOp#lstm_cell_76/BiasAdd/ReadVariableOp2H
"lstm_cell_76/MatMul/ReadVariableOp"lstm_cell_76/MatMul/ReadVariableOp2L
$lstm_cell_76/MatMul_1/ReadVariableOp$lstm_cell_76/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
…	
ч
F__inference_dense_57_layer_call_and_return_conditional_losses_22141752

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
Ќ
while_cond_22142821
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22142821___redundant_placeholder06
2while_while_cond_22142821___redundant_placeholder16
2while_while_cond_22142821___redundant_placeholder26
2while_while_cond_22142821___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Э9
ћ
while_body_22141878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_76_matmul_readvariableop_resource_0:G
5while_lstm_cell_76_matmul_1_readvariableop_resource_0:B
4while_lstm_cell_76_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_76_matmul_readvariableop_resource:E
3while_lstm_cell_76_matmul_1_readvariableop_resource:@
2while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ)while/lstm_cell_76/BiasAdd/ReadVariableOpҐ(while/lstm_cell_76/MatMul/ReadVariableOpҐ*while/lstm_cell_76/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ь
(while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0є
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
*while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0†
while/lstm_cell_76/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
while/lstm_cell_76/addAddV2#while/lstm_cell_76/MatMul:product:0%while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
)while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¶
while/lstm_cell_76/BiasAddBiasAddwhile/lstm_cell_76/add:z:01while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0#while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitz
while/lstm_cell_76/SigmoidSigmoid!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_1Sigmoid!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell_76/mulMul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€t
while/lstm_cell_76/ReluRelu!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell_76/mul_1Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Н
while/lstm_cell_76/add_1AddV2while/lstm_cell_76/mul:z:0while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_2Sigmoid!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell_76/mul_2Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : н
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_76/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y
while/Identity_5Identitywhile/lstm_cell_76/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€–

while/NoOpNoOp*^while/lstm_cell_76/BiasAdd/ReadVariableOp)^while/lstm_cell_76/MatMul/ReadVariableOp+^while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_76_biasadd_readvariableop_resource4while_lstm_cell_76_biasadd_readvariableop_resource_0"l
3while_lstm_cell_76_matmul_1_readvariableop_resource5while_lstm_cell_76_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_76_matmul_readvariableop_resource3while_lstm_cell_76_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2V
)while/lstm_cell_76/BiasAdd/ReadVariableOp)while/lstm_cell_76/BiasAdd/ReadVariableOp2T
(while/lstm_cell_76/MatMul/ReadVariableOp(while/lstm_cell_76/MatMul/ReadVariableOp2X
*while/lstm_cell_76/MatMul_1/ReadVariableOp*while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
џ
f
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141740

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
н
ч
0__inference_sequential_59_layer_call_fn_22141772
lstm_75_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCalllstm_75_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_59_layer_call_and_return_conditional_losses_22141759o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€
'
_user_specified_namelstm_75_input
Ў
р
0__inference_sequential_59_layer_call_fn_22142102

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_59_layer_call_and_return_conditional_losses_22141759o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•K
Ъ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141727

inputs=
+lstm_cell_76_matmul_readvariableop_resource:?
-lstm_cell_76_matmul_1_readvariableop_resource::
,lstm_cell_76_biasadd_readvariableop_resource:
identityИҐ#lstm_cell_76/BiasAdd/ReadVariableOpҐ"lstm_cell_76/MatMul/ReadVariableOpҐ$lstm_cell_76/MatMul_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskО
"lstm_cell_76/MatMul/ReadVariableOpReadVariableOp+lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Х
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0*lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
$lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_cell_76/MatMul_1MatMulzeros:output:0,lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Л
lstm_cell_76/addAddV2lstm_cell_76/MatMul:product:0lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€М
#lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
lstm_cell_76/BiasAddBiasAddlstm_cell_76/add:z:0+lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitn
lstm_cell_76/SigmoidSigmoidlstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€w
lstm_cell_76/mulMullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm_cell_76/ReluRelulstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell_76/mul_1Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm_cell_76/add_1AddV2lstm_cell_76/mul:z:0lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell_76/Relu_1Relulstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell_76/mul_2Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_76_matmul_readvariableop_resource-lstm_cell_76_matmul_1_readvariableop_resource,lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22141642*
condR
while_cond_22141641*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ј
NoOpNoOp$^lstm_cell_76/BiasAdd/ReadVariableOp#^lstm_cell_76/MatMul/ReadVariableOp%^lstm_cell_76/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2J
#lstm_cell_76/BiasAdd/ReadVariableOp#lstm_cell_76/BiasAdd/ReadVariableOp2H
"lstm_cell_76/MatMul/ReadVariableOp"lstm_cell_76/MatMul/ReadVariableOp2L
$lstm_cell_76/MatMul_1/ReadVariableOp$lstm_cell_76/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ш
і
*__inference_lstm_75_layer_call_fn_22142461

inputs
unknown:
	unknown_0:
	unknown_1:
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ю$
л
while_body_22141304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_76_22141328_0:/
while_lstm_cell_76_22141330_0:+
while_lstm_cell_76_22141332_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_76_22141328:-
while_lstm_cell_76_22141330:)
while_lstm_cell_76_22141332:ИҐ*while/lstm_cell_76/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ї
*while/lstm_cell_76/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_76_22141328_0while_lstm_cell_76_22141330_0while_lstm_cell_76_22141332_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141289r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Д
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_76/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Р
while/Identity_4Identity3while/lstm_cell_76/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€Р
while/Identity_5Identity3while/lstm_cell_76/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y

while/NoOpNoOp+^while/lstm_cell_76/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_76_22141328while_lstm_cell_76_22141328_0"<
while_lstm_cell_76_22141330while_lstm_cell_76_22141330_0"<
while_lstm_cell_76_22141332while_lstm_cell_76_22141332_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2X
*while/lstm_cell_76/StatefulPartitionedCall*while/lstm_cell_76/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
њ
Ќ
while_cond_22141641
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22141641___redundant_placeholder06
2while_while_cond_22141641___redundant_placeholder16
2while_while_cond_22141641___redundant_placeholder26
2while_while_cond_22141641___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Ќ
Г
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22141437

inputs

states
states_10
matmul_readvariableop_resource:2
 matmul_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€N
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namestates
ч
f
-__inference_dropout_40_layer_call_fn_22143062

inputs
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141802o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ї
н
&__inference_signature_wrapper_22142087
lstm_75_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCalllstm_75_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_22141222o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€
'
_user_specified_namelstm_75_input
џ
f
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143067

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ёB
ћ

lstm_75_while_body_22142329,
(lstm_75_while_lstm_75_while_loop_counter2
.lstm_75_while_lstm_75_while_maximum_iterations
lstm_75_while_placeholder
lstm_75_while_placeholder_1
lstm_75_while_placeholder_2
lstm_75_while_placeholder_3+
'lstm_75_while_lstm_75_strided_slice_1_0g
clstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0M
;lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0:O
=lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0:J
<lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0:
lstm_75_while_identity
lstm_75_while_identity_1
lstm_75_while_identity_2
lstm_75_while_identity_3
lstm_75_while_identity_4
lstm_75_while_identity_5)
%lstm_75_while_lstm_75_strided_slice_1e
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorK
9lstm_75_while_lstm_cell_76_matmul_readvariableop_resource:M
;lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource:H
:lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOpҐ0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOpҐ2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOpР
?lstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ќ
1lstm_75/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0lstm_75_while_placeholderHlstm_75/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0ђ
0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp;lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0—
!lstm_75/while/lstm_cell_76/MatMulMatMul8lstm_75/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€∞
2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp=lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Є
#lstm_75/while/lstm_cell_76/MatMul_1MatMullstm_75_while_placeholder_2:lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€µ
lstm_75/while/lstm_cell_76/addAddV2+lstm_75/while/lstm_cell_76/MatMul:product:0-lstm_75/while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€™
1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp<lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Њ
"lstm_75/while/lstm_cell_76/BiasAddBiasAdd"lstm_75/while/lstm_cell_76/add:z:09lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€l
*lstm_75/while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :З
 lstm_75/while/lstm_cell_76/splitSplit3lstm_75/while/lstm_cell_76/split/split_dim:output:0+lstm_75/while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitК
"lstm_75/while/lstm_cell_76/SigmoidSigmoid)lstm_75/while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€М
$lstm_75/while/lstm_cell_76/Sigmoid_1Sigmoid)lstm_75/while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ю
lstm_75/while/lstm_cell_76/mulMul(lstm_75/while/lstm_cell_76/Sigmoid_1:y:0lstm_75_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€Д
lstm_75/while/lstm_cell_76/ReluRelu)lstm_75/while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€∞
 lstm_75/while/lstm_cell_76/mul_1Mul&lstm_75/while/lstm_cell_76/Sigmoid:y:0-lstm_75/while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€•
 lstm_75/while/lstm_cell_76/add_1AddV2"lstm_75/while/lstm_cell_76/mul:z:0$lstm_75/while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€М
$lstm_75/while/lstm_cell_76/Sigmoid_2Sigmoid)lstm_75/while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€Б
!lstm_75/while/lstm_cell_76/Relu_1Relu$lstm_75/while/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€і
 lstm_75/while/lstm_cell_76/mul_2Mul(lstm_75/while/lstm_cell_76/Sigmoid_2:y:0/lstm_75/while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€z
8lstm_75/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Н
2lstm_75/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_75_while_placeholder_1Alstm_75/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_75/while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“U
lstm_75/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_75/while/addAddV2lstm_75_while_placeholderlstm_75/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_75/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :З
lstm_75/while/add_1AddV2(lstm_75_while_lstm_75_while_loop_counterlstm_75/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_75/while/IdentityIdentitylstm_75/while/add_1:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: К
lstm_75/while/Identity_1Identity.lstm_75_while_lstm_75_while_maximum_iterations^lstm_75/while/NoOp*
T0*
_output_shapes
: q
lstm_75/while/Identity_2Identitylstm_75/while/add:z:0^lstm_75/while/NoOp*
T0*
_output_shapes
: Ю
lstm_75/while/Identity_3IdentityBlstm_75/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_75/while/NoOp*
T0*
_output_shapes
: С
lstm_75/while/Identity_4Identity$lstm_75/while/lstm_cell_76/mul_2:z:0^lstm_75/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€С
lstm_75/while/Identity_5Identity$lstm_75/while/lstm_cell_76/add_1:z:0^lstm_75/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€р
lstm_75/while/NoOpNoOp2^lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp1^lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp3^lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_75_while_identitylstm_75/while/Identity:output:0"=
lstm_75_while_identity_1!lstm_75/while/Identity_1:output:0"=
lstm_75_while_identity_2!lstm_75/while/Identity_2:output:0"=
lstm_75_while_identity_3!lstm_75/while/Identity_3:output:0"=
lstm_75_while_identity_4!lstm_75/while/Identity_4:output:0"=
lstm_75_while_identity_5!lstm_75/while/Identity_5:output:0"P
%lstm_75_while_lstm_75_strided_slice_1'lstm_75_while_lstm_75_strided_slice_1_0"z
:lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource<lstm_75_while_lstm_cell_76_biasadd_readvariableop_resource_0"|
;lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource=lstm_75_while_lstm_cell_76_matmul_1_readvariableop_resource_0"x
9lstm_75_while_lstm_cell_76_matmul_readvariableop_resource;lstm_75_while_lstm_cell_76_matmul_readvariableop_resource_0"»
alstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensorclstm_75_while_tensorarrayv2read_tensorlistgetitem_lstm_75_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2f
1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp1lstm_75/while/lstm_cell_76/BiasAdd/ReadVariableOp2d
0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp0lstm_75/while/lstm_cell_76/MatMul/ReadVariableOp2h
2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp2lstm_75/while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
њ
Ќ
while_cond_22142676
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22142676___redundant_placeholder06
2while_while_cond_22142676___redundant_placeholder16
2while_while_cond_22142676___redundant_placeholder26
2while_while_cond_22142676___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
О

g
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141802

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•K
Ъ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142907

inputs=
+lstm_cell_76_matmul_readvariableop_resource:?
-lstm_cell_76_matmul_1_readvariableop_resource::
,lstm_cell_76_biasadd_readvariableop_resource:
identityИҐ#lstm_cell_76/BiasAdd/ReadVariableOpҐ"lstm_cell_76/MatMul/ReadVariableOpҐ$lstm_cell_76/MatMul_1/ReadVariableOpҐwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskО
"lstm_cell_76/MatMul/ReadVariableOpReadVariableOp+lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Х
lstm_cell_76/MatMulMatMulstrided_slice_2:output:0*lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
$lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_cell_76/MatMul_1MatMulzeros:output:0,lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Л
lstm_cell_76/addAddV2lstm_cell_76/MatMul:product:0lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€М
#lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
lstm_cell_76/BiasAddBiasAddlstm_cell_76/add:z:0+lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€^
lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё
lstm_cell_76/splitSplit%lstm_cell_76/split/split_dim:output:0lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitn
lstm_cell_76/SigmoidSigmoidlstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_1Sigmoidlstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€w
lstm_cell_76/mulMullstm_cell_76/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
lstm_cell_76/ReluRelulstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ж
lstm_cell_76/mul_1Mullstm_cell_76/Sigmoid:y:0lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€{
lstm_cell_76/add_1AddV2lstm_cell_76/mul:z:0lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€p
lstm_cell_76/Sigmoid_2Sigmoidlstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€e
lstm_cell_76/Relu_1Relulstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€К
lstm_cell_76/mul_2Mullstm_cell_76/Sigmoid_2:y:0!lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :≈
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_76_matmul_readvariableop_resource-lstm_cell_76_matmul_1_readvariableop_resource,lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22142822*
condR
while_cond_22142821*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ÷
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ц
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ј
NoOpNoOp$^lstm_cell_76/BiasAdd/ReadVariableOp#^lstm_cell_76/MatMul/ReadVariableOp%^lstm_cell_76/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2J
#lstm_cell_76/BiasAdd/ReadVariableOp#lstm_cell_76/BiasAdd/ReadVariableOp2H
"lstm_cell_76/MatMul/ReadVariableOp"lstm_cell_76/MatMul/ReadVariableOp2L
$lstm_cell_76/MatMul_1/ReadVariableOp$lstm_cell_76/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•
I
-__inference_dropout_40_layer_call_fn_22143057

inputs
identity≥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141740`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э9
ћ
while_body_22142967
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
3while_lstm_cell_76_matmul_readvariableop_resource_0:G
5while_lstm_cell_76_matmul_1_readvariableop_resource_0:B
4while_lstm_cell_76_biasadd_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
1while_lstm_cell_76_matmul_readvariableop_resource:E
3while_lstm_cell_76_matmul_1_readvariableop_resource:@
2while_lstm_cell_76_biasadd_readvariableop_resource:ИҐ)while/lstm_cell_76/BiasAdd/ReadVariableOpҐ(while/lstm_cell_76/MatMul/ReadVariableOpҐ*while/lstm_cell_76/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ь
(while/lstm_cell_76/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_76_matmul_readvariableop_resource_0*
_output_shapes

:*
dtype0є
while/lstm_cell_76/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
*while/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_76_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0†
while/lstm_cell_76/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Э
while/lstm_cell_76/addAddV2#while/lstm_cell_76/MatMul:product:0%while/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
)while/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_76_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0¶
while/lstm_cell_76/BiasAddBiasAddwhile/lstm_cell_76/add:z:01while/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d
"while/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
while/lstm_cell_76/splitSplit+while/lstm_cell_76/split/split_dim:output:0#while/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitz
while/lstm_cell_76/SigmoidSigmoid!while/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_1Sigmoid!while/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€Ж
while/lstm_cell_76/mulMul while/lstm_cell_76/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€t
while/lstm_cell_76/ReluRelu!while/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€Ш
while/lstm_cell_76/mul_1Mulwhile/lstm_cell_76/Sigmoid:y:0%while/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Н
while/lstm_cell_76/add_1AddV2while/lstm_cell_76/mul:z:0while/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€|
while/lstm_cell_76/Sigmoid_2Sigmoid!while/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€q
while/lstm_cell_76/Relu_1Reluwhile/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
while/lstm_cell_76/mul_2Mul while/lstm_cell_76/Sigmoid_2:y:0'while/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : н
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_76/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_76/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€y
while/Identity_5Identitywhile/lstm_cell_76/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€–

while/NoOpNoOp*^while/lstm_cell_76/BiasAdd/ReadVariableOp)^while/lstm_cell_76/MatMul/ReadVariableOp+^while/lstm_cell_76/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_76_biasadd_readvariableop_resource4while_lstm_cell_76_biasadd_readvariableop_resource_0"l
3while_lstm_cell_76_matmul_1_readvariableop_resource5while_lstm_cell_76_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_76_matmul_readvariableop_resource3while_lstm_cell_76_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : 2V
)while/lstm_cell_76/BiasAdd/ReadVariableOp)while/lstm_cell_76/BiasAdd/ReadVariableOp2T
(while/lstm_cell_76/MatMul/ReadVariableOp(while/lstm_cell_76/MatMul/ReadVariableOp2X
*while/lstm_cell_76/MatMul_1/ReadVariableOp*while/lstm_cell_76/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: 
њ
Ќ
while_cond_22142531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22142531___redundant_placeholder06
2while_while_cond_22142531___redundant_placeholder16
2while_while_cond_22142531___redundant_placeholder26
2while_while_cond_22142531___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
њ
Ќ
while_cond_22141877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22141877___redundant_placeholder06
2while_while_cond_22141877___redundant_placeholder16
2while_while_cond_22141877___redundant_placeholder26
2while_while_cond_22141877___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€:€€€€€€€€€: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
:
Иq
З
#__inference__wrapped_model_22141222
lstm_75_inputS
Asequential_59_lstm_75_lstm_cell_76_matmul_readvariableop_resource:U
Csequential_59_lstm_75_lstm_cell_76_matmul_1_readvariableop_resource:P
Bsequential_59_lstm_75_lstm_cell_76_biasadd_readvariableop_resource:G
5sequential_59_dense_57_matmul_readvariableop_resource:D
6sequential_59_dense_57_biasadd_readvariableop_resource:
identityИҐ-sequential_59/dense_57/BiasAdd/ReadVariableOpҐ,sequential_59/dense_57/MatMul/ReadVariableOpҐ9sequential_59/lstm_75/lstm_cell_76/BiasAdd/ReadVariableOpҐ8sequential_59/lstm_75/lstm_cell_76/MatMul/ReadVariableOpҐ:sequential_59/lstm_75/lstm_cell_76/MatMul_1/ReadVariableOpҐsequential_59/lstm_75/whileX
sequential_59/lstm_75/ShapeShapelstm_75_input*
T0*
_output_shapes
:s
)sequential_59/lstm_75/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_59/lstm_75/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_59/lstm_75/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
#sequential_59/lstm_75/strided_sliceStridedSlice$sequential_59/lstm_75/Shape:output:02sequential_59/lstm_75/strided_slice/stack:output:04sequential_59/lstm_75/strided_slice/stack_1:output:04sequential_59/lstm_75/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_59/lstm_75/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :µ
"sequential_59/lstm_75/zeros/packedPack,sequential_59/lstm_75/strided_slice:output:0-sequential_59/lstm_75/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_59/lstm_75/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ѓ
sequential_59/lstm_75/zerosFill+sequential_59/lstm_75/zeros/packed:output:0*sequential_59/lstm_75/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
&sequential_59/lstm_75/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :є
$sequential_59/lstm_75/zeros_1/packedPack,sequential_59/lstm_75/strided_slice:output:0/sequential_59/lstm_75/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_59/lstm_75/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
sequential_59/lstm_75/zeros_1Fill-sequential_59/lstm_75/zeros_1/packed:output:0,sequential_59/lstm_75/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€y
$sequential_59/lstm_75/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
sequential_59/lstm_75/transpose	Transposelstm_75_input-sequential_59/lstm_75/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€p
sequential_59/lstm_75/Shape_1Shape#sequential_59/lstm_75/transpose:y:0*
T0*
_output_shapes
:u
+sequential_59/lstm_75/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_59/lstm_75/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_59/lstm_75/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%sequential_59/lstm_75/strided_slice_1StridedSlice&sequential_59/lstm_75/Shape_1:output:04sequential_59/lstm_75/strided_slice_1/stack:output:06sequential_59/lstm_75/strided_slice_1/stack_1:output:06sequential_59/lstm_75/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_59/lstm_75/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ц
#sequential_59/lstm_75/TensorArrayV2TensorListReserve:sequential_59/lstm_75/TensorArrayV2/element_shape:output:0.sequential_59/lstm_75/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ь
Ksequential_59/lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ґ
=sequential_59/lstm_75/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_59/lstm_75/transpose:y:0Tsequential_59/lstm_75/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“u
+sequential_59/lstm_75/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_59/lstm_75/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_59/lstm_75/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:„
%sequential_59/lstm_75/strided_slice_2StridedSlice#sequential_59/lstm_75/transpose:y:04sequential_59/lstm_75/strided_slice_2/stack:output:06sequential_59/lstm_75/strided_slice_2/stack_1:output:06sequential_59/lstm_75/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskЇ
8sequential_59/lstm_75/lstm_cell_76/MatMul/ReadVariableOpReadVariableOpAsequential_59_lstm_75_lstm_cell_76_matmul_readvariableop_resource*
_output_shapes

:*
dtype0„
)sequential_59/lstm_75/lstm_cell_76/MatMulMatMul.sequential_59/lstm_75/strided_slice_2:output:0@sequential_59/lstm_75/lstm_cell_76/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Њ
:sequential_59/lstm_75/lstm_cell_76/MatMul_1/ReadVariableOpReadVariableOpCsequential_59_lstm_75_lstm_cell_76_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0—
+sequential_59/lstm_75/lstm_cell_76/MatMul_1MatMul$sequential_59/lstm_75/zeros:output:0Bsequential_59/lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ќ
&sequential_59/lstm_75/lstm_cell_76/addAddV23sequential_59/lstm_75/lstm_cell_76/MatMul:product:05sequential_59/lstm_75/lstm_cell_76/MatMul_1:product:0*
T0*'
_output_shapes
:€€€€€€€€€Є
9sequential_59/lstm_75/lstm_cell_76/BiasAdd/ReadVariableOpReadVariableOpBsequential_59_lstm_75_lstm_cell_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0÷
*sequential_59/lstm_75/lstm_cell_76/BiasAddBiasAdd*sequential_59/lstm_75/lstm_cell_76/add:z:0Asequential_59/lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€t
2sequential_59/lstm_75/lstm_cell_76/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Я
(sequential_59/lstm_75/lstm_cell_76/splitSplit;sequential_59/lstm_75/lstm_cell_76/split/split_dim:output:03sequential_59/lstm_75/lstm_cell_76/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*
	num_splitЪ
*sequential_59/lstm_75/lstm_cell_76/SigmoidSigmoid1sequential_59/lstm_75/lstm_cell_76/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
,sequential_59/lstm_75/lstm_cell_76/Sigmoid_1Sigmoid1sequential_59/lstm_75/lstm_cell_76/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€є
&sequential_59/lstm_75/lstm_cell_76/mulMul0sequential_59/lstm_75/lstm_cell_76/Sigmoid_1:y:0&sequential_59/lstm_75/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
'sequential_59/lstm_75/lstm_cell_76/ReluRelu1sequential_59/lstm_75/lstm_cell_76/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€»
(sequential_59/lstm_75/lstm_cell_76/mul_1Mul.sequential_59/lstm_75/lstm_cell_76/Sigmoid:y:05sequential_59/lstm_75/lstm_cell_76/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€љ
(sequential_59/lstm_75/lstm_cell_76/add_1AddV2*sequential_59/lstm_75/lstm_cell_76/mul:z:0,sequential_59/lstm_75/lstm_cell_76/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€Ь
,sequential_59/lstm_75/lstm_cell_76/Sigmoid_2Sigmoid1sequential_59/lstm_75/lstm_cell_76/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€С
)sequential_59/lstm_75/lstm_cell_76/Relu_1Relu,sequential_59/lstm_75/lstm_cell_76/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ћ
(sequential_59/lstm_75/lstm_cell_76/mul_2Mul0sequential_59/lstm_75/lstm_cell_76/Sigmoid_2:y:07sequential_59/lstm_75/lstm_cell_76/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€Д
3sequential_59/lstm_75/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   t
2sequential_59/lstm_75/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :З
%sequential_59/lstm_75/TensorArrayV2_1TensorListReserve<sequential_59/lstm_75/TensorArrayV2_1/element_shape:output:0;sequential_59/lstm_75/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“\
sequential_59/lstm_75/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_59/lstm_75/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€j
(sequential_59/lstm_75/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ї
sequential_59/lstm_75/whileWhile1sequential_59/lstm_75/while/loop_counter:output:07sequential_59/lstm_75/while/maximum_iterations:output:0#sequential_59/lstm_75/time:output:0.sequential_59/lstm_75/TensorArrayV2_1:handle:0$sequential_59/lstm_75/zeros:output:0&sequential_59/lstm_75/zeros_1:output:0.sequential_59/lstm_75/strided_slice_1:output:0Msequential_59/lstm_75/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_59_lstm_75_lstm_cell_76_matmul_readvariableop_resourceCsequential_59_lstm_75_lstm_cell_76_matmul_1_readvariableop_resourceBsequential_59_lstm_75_lstm_cell_76_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_59_lstm_75_while_body_22141130*5
cond-R+
)sequential_59_lstm_75_while_cond_22141129*K
output_shapes:
8: : : : :€€€€€€€€€:€€€€€€€€€: : : : : *
parallel_iterations Ч
Fsequential_59/lstm_75/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ш
8sequential_59/lstm_75/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_59/lstm_75/while:output:3Osequential_59/lstm_75/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€*
element_dtype0*
num_elements~
+sequential_59/lstm_75/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€w
-sequential_59/lstm_75/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_59/lstm_75/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
%sequential_59/lstm_75/strided_slice_3StridedSliceAsequential_59/lstm_75/TensorArrayV2Stack/TensorListStack:tensor:04sequential_59/lstm_75/strided_slice_3/stack:output:06sequential_59/lstm_75/strided_slice_3/stack_1:output:06sequential_59/lstm_75/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask{
&sequential_59/lstm_75/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ў
!sequential_59/lstm_75/transpose_1	TransposeAsequential_59/lstm_75/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_59/lstm_75/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€q
sequential_59/lstm_75/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    П
!sequential_59/dropout_40/IdentityIdentity.sequential_59/lstm_75/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ґ
,sequential_59/dense_57/MatMul/ReadVariableOpReadVariableOp5sequential_59_dense_57_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ї
sequential_59/dense_57/MatMulMatMul*sequential_59/dropout_40/Identity:output:04sequential_59/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
-sequential_59/dense_57/BiasAdd/ReadVariableOpReadVariableOp6sequential_59_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ї
sequential_59/dense_57/BiasAddBiasAdd'sequential_59/dense_57/MatMul:product:05sequential_59/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
IdentityIdentity'sequential_59/dense_57/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ч
NoOpNoOp.^sequential_59/dense_57/BiasAdd/ReadVariableOp-^sequential_59/dense_57/MatMul/ReadVariableOp:^sequential_59/lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp9^sequential_59/lstm_75/lstm_cell_76/MatMul/ReadVariableOp;^sequential_59/lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp^sequential_59/lstm_75/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 2^
-sequential_59/dense_57/BiasAdd/ReadVariableOp-sequential_59/dense_57/BiasAdd/ReadVariableOp2\
,sequential_59/dense_57/MatMul/ReadVariableOp,sequential_59/dense_57/MatMul/ReadVariableOp2v
9sequential_59/lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp9sequential_59/lstm_75/lstm_cell_76/BiasAdd/ReadVariableOp2t
8sequential_59/lstm_75/lstm_cell_76/MatMul/ReadVariableOp8sequential_59/lstm_75/lstm_cell_76/MatMul/ReadVariableOp2x
:sequential_59/lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp:sequential_59/lstm_75/lstm_cell_76/MatMul_1/ReadVariableOp2:
sequential_59/lstm_75/whilesequential_59/lstm_75/while:Z V
+
_output_shapes
:€€€€€€€€€
'
_user_specified_namelstm_75_input
О

g
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143079

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>¶
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ў
р
0__inference_sequential_59_layer_call_fn_22142117

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142006o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ё
д
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142051
lstm_75_input"
lstm_75_22142037:"
lstm_75_22142039:
lstm_75_22142041:#
dense_57_22142045:
dense_57_22142047:
identityИҐ dense_57/StatefulPartitionedCallҐlstm_75/StatefulPartitionedCallН
lstm_75/StatefulPartitionedCallStatefulPartitionedCalllstm_75_inputlstm_75_22142037lstm_75_22142039lstm_75_22142041*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_75_layer_call_and_return_conditional_losses_22141727а
dropout_40/PartitionedCallPartitionedCall(lstm_75/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22141740У
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#dropout_40/PartitionedCall:output:0dense_57_22142045dense_57_22142047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_57_layer_call_and_return_conditional_losses_22141752x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Л
NoOpNoOp!^dense_57/StatefulPartitionedCall ^lstm_75/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€: : : : : 2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2B
lstm_75/StatefulPartitionedCalllstm_75/StatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€
'
_user_specified_namelstm_75_input"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ї
serving_defaultІ
K
lstm_75_input:
serving_default_lstm_75_input:0€€€€€€€€€<
dense_570
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:яє
Ѕ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Џ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
Љ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
ї
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
C
%0
&1
'2
#3
$4"
trackable_list_wrapper
C
%0
&1
'2
#3
$4"
trackable_list_wrapper
 "
trackable_list_wrapper
 
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
х
-trace_0
.trace_1
/trace_2
0trace_32К
0__inference_sequential_59_layer_call_fn_22141772
0__inference_sequential_59_layer_call_fn_22142102
0__inference_sequential_59_layer_call_fn_22142117
0__inference_sequential_59_layer_call_fn_22142034њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z-trace_0z.trace_1z/trace_2z0trace_3
б
1trace_0
2trace_1
3trace_2
4trace_32ц
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142269
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142428
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142051
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142068њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z1trace_0z2trace_1z3trace_2z4trace_3
‘B—
#__inference__wrapped_model_22141222lstm_75_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ь
5
_variables
6_iterations
7_learning_rate
8_index_dict
9
_momentums
:_velocities
;_update_step_xla"
experimentalOptimizer
,
<serving_default"
signature_map
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
є

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
т
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32З
*__inference_lstm_75_layer_call_fn_22142439
*__inference_lstm_75_layer_call_fn_22142450
*__inference_lstm_75_layer_call_fn_22142461
*__inference_lstm_75_layer_call_fn_22142472‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
ё
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32у
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142617
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142762
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142907
E__inference_lstm_75_layer_call_and_return_conditional_losses_22143052‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
"
_generic_user_object
ш
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses
Q_random_generator
R
state_size

%kernel
&recurrent_kernel
'bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ћ
Xtrace_0
Ytrace_12Ф
-__inference_dropout_40_layer_call_fn_22143057
-__inference_dropout_40_layer_call_fn_22143062≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zXtrace_0zYtrace_1
Б
Ztrace_0
[trace_12 
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143067
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143079≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zZtrace_0z[trace_1
"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
п
atrace_02“
+__inference_dense_57_layer_call_fn_22143088Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zatrace_0
К
btrace_02н
F__inference_dense_57_layer_call_and_return_conditional_losses_22143098Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zbtrace_0
!:2dense_57/kernel
:2dense_57/bias
-:+2lstm_75/lstm_cell_76/kernel
7:52%lstm_75/lstm_cell_76/recurrent_kernel
':%2lstm_75/lstm_cell_76/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ИBЕ
0__inference_sequential_59_layer_call_fn_22141772lstm_75_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
0__inference_sequential_59_layer_call_fn_22142102inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
БBю
0__inference_sequential_59_layer_call_fn_22142117inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ИBЕ
0__inference_sequential_59_layer_call_fn_22142034lstm_75_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЬBЩ
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142269inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЬBЩ
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142428inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
£B†
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142051lstm_75_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
£B†
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142068lstm_75_input"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
n
60
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
C
e0
g1
i2
k3
m4"
trackable_list_wrapper
C
f0
h1
j2
l3
n4"
trackable_list_wrapper
њ2Љє
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
”B–
&__inference_signature_wrapper_22142087lstm_75_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBП
*__inference_lstm_75_layer_call_fn_22142439inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ТBП
*__inference_lstm_75_layer_call_fn_22142450inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
РBН
*__inference_lstm_75_layer_call_fn_22142461inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
РBН
*__inference_lstm_75_layer_call_fn_22142472inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠B™
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142617inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠B™
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142762inputs_0"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЂB®
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142907inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЂB®
E__inference_lstm_75_layer_call_and_return_conditional_losses_22143052inputs"‘
Ћ≤«
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
5
%0
&1
'2"
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
ў
ttrace_0
utrace_12Ґ
/__inference_lstm_cell_76_layer_call_fn_22143115
/__inference_lstm_cell_76_layer_call_fn_22143132љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zttrace_0zutrace_1
П
vtrace_0
wtrace_12Ў
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143164
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143196љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zvtrace_0zwtrace_1
"
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
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
-__inference_dropout_40_layer_call_fn_22143057inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
тBп
-__inference_dropout_40_layer_call_fn_22143062inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143067inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143079inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
яB№
+__inference_dense_57_layer_call_fn_22143088inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_dense_57_layer_call_and_return_conditional_losses_22143098inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
N
x	variables
y	keras_api
	ztotal
	{count"
_tf_keras_metric
_
|	variables
}	keras_api
	~total
	count
А
_fn_kwargs"
_tf_keras_metric
2:02"Adam/m/lstm_75/lstm_cell_76/kernel
2:02"Adam/v/lstm_75/lstm_cell_76/kernel
<::2,Adam/m/lstm_75/lstm_cell_76/recurrent_kernel
<::2,Adam/v/lstm_75/lstm_cell_76/recurrent_kernel
,:*2 Adam/m/lstm_75/lstm_cell_76/bias
,:*2 Adam/v/lstm_75/lstm_cell_76/bias
&:$2Adam/m/dense_57/kernel
&:$2Adam/v/dense_57/kernel
 :2Adam/m/dense_57/bias
 :2Adam/v/dense_57/bias
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
ТBП
/__inference_lstm_cell_76_layer_call_fn_22143115inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ТBП
/__inference_lstm_cell_76_layer_call_fn_22143132inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠B™
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143164inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠B™
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143196inputsstates_0states_1"љ
і≤∞
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
z0
{1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЯ
#__inference__wrapped_model_22141222x%&'#$:Ґ7
0Ґ-
+К(
lstm_75_input€€€€€€€€€
™ "3™0
.
dense_57"К
dense_57€€€€€€€€€≠
F__inference_dense_57_layer_call_and_return_conditional_losses_22143098c#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ З
+__inference_dense_57_layer_call_fn_22143088X#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€ѓ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143067c3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ѓ
H__inference_dropout_40_layer_call_and_return_conditional_losses_22143079c3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Й
-__inference_dropout_40_layer_call_fn_22143057X3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "!К
unknown€€€€€€€€€Й
-__inference_dropout_40_layer_call_fn_22143062X3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "!К
unknown€€€€€€€€€ќ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142617Д%&'OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ќ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142762Д%&'OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ љ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22142907t%&'?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ љ
E__inference_lstm_75_layer_call_and_return_conditional_losses_22143052t%&'?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ І
*__inference_lstm_75_layer_call_fn_22142439y%&'OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "!К
unknown€€€€€€€€€І
*__inference_lstm_75_layer_call_fn_22142450y%&'OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ "!К
unknown€€€€€€€€€Ч
*__inference_lstm_75_layer_call_fn_22142461i%&'?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "!К
unknown€€€€€€€€€Ч
*__inference_lstm_75_layer_call_fn_22142472i%&'?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "!К
unknown€€€€€€€€€г
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143164Ф%&'АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p 
™ "ЙҐЕ
~Ґ{
$К!

tensor_0_0€€€€€€€€€
SЪP
&К#
tensor_0_1_0€€€€€€€€€
&К#
tensor_0_1_1€€€€€€€€€
Ъ г
J__inference_lstm_cell_76_layer_call_and_return_conditional_losses_22143196Ф%&'АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p
™ "ЙҐЕ
~Ґ{
$К!

tensor_0_0€€€€€€€€€
SЪP
&К#
tensor_0_1_0€€€€€€€€€
&К#
tensor_0_1_1€€€€€€€€€
Ъ ґ
/__inference_lstm_cell_76_layer_call_fn_22143115В%&'АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p 
™ "xҐu
"К
tensor_0€€€€€€€€€
OЪL
$К!

tensor_1_0€€€€€€€€€
$К!

tensor_1_1€€€€€€€€€ґ
/__inference_lstm_cell_76_layer_call_fn_22143132В%&'АҐ}
vҐs
 К
inputs€€€€€€€€€
KҐH
"К
states_0€€€€€€€€€
"К
states_1€€€€€€€€€
p
™ "xҐu
"К
tensor_0€€€€€€€€€
OЪL
$К!

tensor_1_0€€€€€€€€€
$К!

tensor_1_1€€€€€€€€€»
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142051y%&'#$BҐ?
8Ґ5
+К(
lstm_75_input€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ »
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142068y%&'#$BҐ?
8Ґ5
+К(
lstm_75_input€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ѕ
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142269r%&'#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ѕ
K__inference_sequential_59_layer_call_and_return_conditional_losses_22142428r%&'#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ґ
0__inference_sequential_59_layer_call_fn_22141772n%&'#$BҐ?
8Ґ5
+К(
lstm_75_input€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€Ґ
0__inference_sequential_59_layer_call_fn_22142034n%&'#$BҐ?
8Ґ5
+К(
lstm_75_input€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€Ы
0__inference_sequential_59_layer_call_fn_22142102g%&'#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€Ы
0__inference_sequential_59_layer_call_fn_22142117g%&'#$;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€і
&__inference_signature_wrapper_22142087Й%&'#$KҐH
Ґ 
A™>
<
lstm_75_input+К(
lstm_75_input€€€€€€€€€"3™0
.
dense_57"К
dense_57€€€€€€€€€