��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
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
�"serve*2.11.02v2.11.0-rc2-15-g6290819256d8��
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
�
Adam/v/dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_61/bias
y
(Adam/v/dense_61/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_61/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_61/bias
y
(Adam/m/dense_61/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_61/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_61/kernel
�
*Adam/v/dense_61/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_61/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_61/kernel
�
*Adam/m/dense_61/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_61/kernel*
_output_shapes
:	�*
dtype0
�
 Adam/v/lstm_79/lstm_cell_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/v/lstm_79/lstm_cell_80/bias
�
4Adam/v/lstm_79/lstm_cell_80/bias/Read/ReadVariableOpReadVariableOp Adam/v/lstm_79/lstm_cell_80/bias*
_output_shapes	
:�*
dtype0
�
 Adam/m/lstm_79/lstm_cell_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/m/lstm_79/lstm_cell_80/bias
�
4Adam/m/lstm_79/lstm_cell_80/bias/Read/ReadVariableOpReadVariableOp Adam/m/lstm_79/lstm_cell_80/bias*
_output_shapes	
:�*
dtype0
�
,Adam/v/lstm_79/lstm_cell_80/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*=
shared_name.,Adam/v/lstm_79/lstm_cell_80/recurrent_kernel
�
@Adam/v/lstm_79/lstm_cell_80/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/v/lstm_79/lstm_cell_80/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
,Adam/m/lstm_79/lstm_cell_80/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*=
shared_name.,Adam/m/lstm_79/lstm_cell_80/recurrent_kernel
�
@Adam/m/lstm_79/lstm_cell_80/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/m/lstm_79/lstm_cell_80/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
"Adam/v/lstm_79/lstm_cell_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/v/lstm_79/lstm_cell_80/kernel
�
6Adam/v/lstm_79/lstm_cell_80/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_79/lstm_cell_80/kernel*
_output_shapes
:	�*
dtype0
�
"Adam/m/lstm_79/lstm_cell_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/m/lstm_79/lstm_cell_80/kernel
�
6Adam/m/lstm_79/lstm_cell_80/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_79/lstm_cell_80/kernel*
_output_shapes
:	�*
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
�
lstm_79/lstm_cell_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_79/lstm_cell_80/bias
�
-lstm_79/lstm_cell_80/bias/Read/ReadVariableOpReadVariableOplstm_79/lstm_cell_80/bias*
_output_shapes	
:�*
dtype0
�
%lstm_79/lstm_cell_80/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*6
shared_name'%lstm_79/lstm_cell_80/recurrent_kernel
�
9lstm_79/lstm_cell_80/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_79/lstm_cell_80/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
lstm_79/lstm_cell_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_79/lstm_cell_80/kernel
�
/lstm_79/lstm_cell_80/kernel/Read/ReadVariableOpReadVariableOplstm_79/lstm_cell_80/kernel*
_output_shapes
:	�*
dtype0
r
dense_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_61/bias
k
!dense_61/bias/Read/ReadVariableOpReadVariableOpdense_61/bias*
_output_shapes
:*
dtype0
{
dense_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_61/kernel
t
#dense_61/kernel/Read/ReadVariableOpReadVariableOpdense_61/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_lstm_79_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_79_inputlstm_79/lstm_cell_80/kernel%lstm_79/lstm_cell_80/recurrent_kernellstm_79/lstm_cell_80/biasdense_61/kerneldense_61/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_22292782

NoOpNoOp
�/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�/
value�/B�/ B�/
�
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
�
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
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
�
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
�
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
�
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
�

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
�
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
�
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
�
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
VARIABLE_VALUEdense_61/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_61/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_79/lstm_cell_80/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_79/lstm_cell_80/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_79/lstm_cell_80/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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
�
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
�
_fn_kwargs*
mg
VARIABLE_VALUE"Adam/m/lstm_79/lstm_cell_80/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_79/lstm_cell_80/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/m/lstm_79/lstm_cell_80/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/v/lstm_79/lstm_cell_80/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_79/lstm_cell_80/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_79/lstm_cell_80/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_61/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_61/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_61/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_61/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
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
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_61/kernel/Read/ReadVariableOp!dense_61/bias/Read/ReadVariableOp/lstm_79/lstm_cell_80/kernel/Read/ReadVariableOp9lstm_79/lstm_cell_80/recurrent_kernel/Read/ReadVariableOp-lstm_79/lstm_cell_80/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/lstm_79/lstm_cell_80/kernel/Read/ReadVariableOp6Adam/v/lstm_79/lstm_cell_80/kernel/Read/ReadVariableOp@Adam/m/lstm_79/lstm_cell_80/recurrent_kernel/Read/ReadVariableOp@Adam/v/lstm_79/lstm_cell_80/recurrent_kernel/Read/ReadVariableOp4Adam/m/lstm_79/lstm_cell_80/bias/Read/ReadVariableOp4Adam/v/lstm_79/lstm_cell_80/bias/Read/ReadVariableOp*Adam/m/dense_61/kernel/Read/ReadVariableOp*Adam/v/dense_61/kernel/Read/ReadVariableOp(Adam/m/dense_61/bias/Read/ReadVariableOp(Adam/v/dense_61/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*"
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
GPU 2J 8� **
f%R#
!__inference__traced_save_22293977
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_61/kerneldense_61/biaslstm_79/lstm_cell_80/kernel%lstm_79/lstm_cell_80/recurrent_kernellstm_79/lstm_cell_80/bias	iterationlearning_rate"Adam/m/lstm_79/lstm_cell_80/kernel"Adam/v/lstm_79/lstm_cell_80/kernel,Adam/m/lstm_79/lstm_cell_80/recurrent_kernel,Adam/v/lstm_79/lstm_cell_80/recurrent_kernel Adam/m/lstm_79/lstm_cell_80/bias Adam/v/lstm_79/lstm_cell_80/biasAdam/m/dense_61/kernelAdam/v/dense_61/kernelAdam/m/dense_61/biasAdam/v/dense_61/biastotal_1count_1totalcount*!
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
GPU 2J 8� *-
f(R&
$__inference__traced_restore_22294050��
�
�
*__inference_lstm_79_layer_call_fn_22293134
inputs_0
unknown:	�
	unknown_0:
��
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292069p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
while_cond_22293516
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22293516___redundant_placeholder06
2while_while_cond_22293516___redundant_placeholder16
2while_while_cond_22293516___redundant_placeholder26
2while_while_cond_22293516___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�C
�

lstm_79_while_body_22293024,
(lstm_79_while_lstm_79_while_loop_counter2
.lstm_79_while_lstm_79_while_maximum_iterations
lstm_79_while_placeholder
lstm_79_while_placeholder_1
lstm_79_while_placeholder_2
lstm_79_while_placeholder_3+
'lstm_79_while_lstm_79_strided_slice_1_0g
clstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0:	�Q
=lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��K
<lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
lstm_79_while_identity
lstm_79_while_identity_1
lstm_79_while_identity_2
lstm_79_while_identity_3
lstm_79_while_identity_4
lstm_79_while_identity_5)
%lstm_79_while_lstm_79_strided_slice_1e
alstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensorL
9lstm_79_while_lstm_cell_80_matmul_readvariableop_resource:	�O
;lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource:
��I
:lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource:	���1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp�0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp�2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp�
?lstm_79/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_79/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensor_0lstm_79_while_placeholderHlstm_79/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp;lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_79/while/lstm_cell_80/MatMulMatMul8lstm_79/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp=lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
#lstm_79/while/lstm_cell_80/MatMul_1MatMullstm_79_while_placeholder_2:lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_79/while/lstm_cell_80/addAddV2+lstm_79/while/lstm_cell_80/MatMul:product:0-lstm_79/while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp<lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_79/while/lstm_cell_80/BiasAddBiasAdd"lstm_79/while/lstm_cell_80/add:z:09lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_79/while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_79/while/lstm_cell_80/splitSplit3lstm_79/while/lstm_cell_80/split/split_dim:output:0+lstm_79/while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
"lstm_79/while/lstm_cell_80/SigmoidSigmoid)lstm_79/while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:�����������
$lstm_79/while/lstm_cell_80/Sigmoid_1Sigmoid)lstm_79/while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
lstm_79/while/lstm_cell_80/mulMul(lstm_79/while/lstm_cell_80/Sigmoid_1:y:0lstm_79_while_placeholder_3*
T0*(
_output_shapes
:�����������
lstm_79/while/lstm_cell_80/ReluRelu)lstm_79/while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
 lstm_79/while/lstm_cell_80/mul_1Mul&lstm_79/while/lstm_cell_80/Sigmoid:y:0-lstm_79/while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
 lstm_79/while/lstm_cell_80/add_1AddV2"lstm_79/while/lstm_cell_80/mul:z:0$lstm_79/while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:�����������
$lstm_79/while/lstm_cell_80/Sigmoid_2Sigmoid)lstm_79/while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:�����������
!lstm_79/while/lstm_cell_80/Relu_1Relu$lstm_79/while/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
 lstm_79/while/lstm_cell_80/mul_2Mul(lstm_79/while/lstm_cell_80/Sigmoid_2:y:0/lstm_79/while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������z
8lstm_79/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_79/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_79_while_placeholder_1Alstm_79/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_79/while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_79/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_79/while/addAddV2lstm_79_while_placeholderlstm_79/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_79/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_79/while/add_1AddV2(lstm_79_while_lstm_79_while_loop_counterlstm_79/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_79/while/IdentityIdentitylstm_79/while/add_1:z:0^lstm_79/while/NoOp*
T0*
_output_shapes
: �
lstm_79/while/Identity_1Identity.lstm_79_while_lstm_79_while_maximum_iterations^lstm_79/while/NoOp*
T0*
_output_shapes
: q
lstm_79/while/Identity_2Identitylstm_79/while/add:z:0^lstm_79/while/NoOp*
T0*
_output_shapes
: �
lstm_79/while/Identity_3IdentityBlstm_79/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_79/while/NoOp*
T0*
_output_shapes
: �
lstm_79/while/Identity_4Identity$lstm_79/while/lstm_cell_80/mul_2:z:0^lstm_79/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_79/while/Identity_5Identity$lstm_79/while/lstm_cell_80/add_1:z:0^lstm_79/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_79/while/NoOpNoOp2^lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp1^lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp3^lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_79_while_identitylstm_79/while/Identity:output:0"=
lstm_79_while_identity_1!lstm_79/while/Identity_1:output:0"=
lstm_79_while_identity_2!lstm_79/while/Identity_2:output:0"=
lstm_79_while_identity_3!lstm_79/while/Identity_3:output:0"=
lstm_79_while_identity_4!lstm_79/while/Identity_4:output:0"=
lstm_79_while_identity_5!lstm_79/while/Identity_5:output:0"P
%lstm_79_while_lstm_79_strided_slice_1'lstm_79_while_lstm_79_strided_slice_1_0"z
:lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource<lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0"|
;lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource=lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0"x
9lstm_79_while_lstm_cell_80_matmul_readvariableop_resource;lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0"�
alstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensorclstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2f
1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp2d
0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp2h
2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
f
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292435

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�9
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292069

inputs(
lstm_cell_80_22291985:	�)
lstm_cell_80_22291987:
��$
lstm_cell_80_22291989:	�
identity��$lstm_cell_80/StatefulPartitionedCall�while;
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_80/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_80_22291985lstm_cell_80_22291987lstm_cell_80_22291989*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22291984n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_80_22291985lstm_cell_80_22291987lstm_cell_80_22291989*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22291999*
condR
while_cond_22291998*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:����������u
NoOpNoOp%^lstm_cell_80/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_80/StatefulPartitionedCall$lstm_cell_80/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
/__inference_lstm_cell_80_layer_call_fn_22293827

inputs
states_0
states_1
unknown:	�
	unknown_0:
��
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22292132p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������:����������:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states_0:RN
(
_output_shapes
:����������
"
_user_specified_name
states_1
�
�
while_cond_22291998
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22291998___redundant_placeholder06
2while_while_cond_22291998___redundant_placeholder16
2while_while_cond_22291998___redundant_placeholder26
2while_while_cond_22291998___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_22292191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22292191___redundant_placeholder06
2while_while_cond_22292191___redundant_placeholder16
2while_while_cond_22292191___redundant_placeholder26
2while_while_cond_22292191___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�$
�
while_body_22291999
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_80_22292023_0:	�1
while_lstm_cell_80_22292025_0:
��,
while_lstm_cell_80_22292027_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_80_22292023:	�/
while_lstm_cell_80_22292025:
��*
while_lstm_cell_80_22292027:	���*while/lstm_cell_80/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_80/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_80_22292023_0while_lstm_cell_80_22292025_0while_lstm_cell_80_22292027_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22291984r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_80/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_80/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:�����������
while/Identity_5Identity3while/lstm_cell_80/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:����������y

while/NoOpNoOp+^while/lstm_cell_80/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_80_22292023while_lstm_cell_80_22292023_0"<
while_lstm_cell_80_22292025while_lstm_cell_80_22292025_0"<
while_lstm_cell_80_22292027while_lstm_cell_80_22292027_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2X
*while/lstm_cell_80/StatefulPartitionedCall*while/lstm_cell_80/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_22293371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22293371___redundant_placeholder06
2while_while_cond_22293371___redundant_placeholder16
2while_while_cond_22293371___redundant_placeholder26
2while_while_cond_22293371___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�e
�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22293123

inputsF
3lstm_79_lstm_cell_80_matmul_readvariableop_resource:	�I
5lstm_79_lstm_cell_80_matmul_1_readvariableop_resource:
��C
4lstm_79_lstm_cell_80_biasadd_readvariableop_resource:	�:
'dense_61_matmul_readvariableop_resource:	�6
(dense_61_biasadd_readvariableop_resource:
identity��dense_61/BiasAdd/ReadVariableOp�dense_61/MatMul/ReadVariableOp�+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp�*lstm_79/lstm_cell_80/MatMul/ReadVariableOp�,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp�lstm_79/whileC
lstm_79/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_79/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_79/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_79/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_sliceStridedSlicelstm_79/Shape:output:0$lstm_79/strided_slice/stack:output:0&lstm_79/strided_slice/stack_1:output:0&lstm_79/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_79/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_79/zeros/packedPacklstm_79/strided_slice:output:0lstm_79/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_79/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_79/zerosFilllstm_79/zeros/packed:output:0lstm_79/zeros/Const:output:0*
T0*(
_output_shapes
:����������[
lstm_79/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_79/zeros_1/packedPacklstm_79/strided_slice:output:0!lstm_79/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_79/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_79/zeros_1Filllstm_79/zeros_1/packed:output:0lstm_79/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������k
lstm_79/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_79/transpose	Transposeinputslstm_79/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_79/Shape_1Shapelstm_79/transpose:y:0*
T0*
_output_shapes
:g
lstm_79/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_79/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_79/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_slice_1StridedSlicelstm_79/Shape_1:output:0&lstm_79/strided_slice_1/stack:output:0(lstm_79/strided_slice_1/stack_1:output:0(lstm_79/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_79/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_79/TensorArrayV2TensorListReserve,lstm_79/TensorArrayV2/element_shape:output:0 lstm_79/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_79/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_79/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_79/transpose:y:0Flstm_79/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_79/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_79/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_79/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_slice_2StridedSlicelstm_79/transpose:y:0&lstm_79/strided_slice_2/stack:output:0(lstm_79/strided_slice_2/stack_1:output:0(lstm_79/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_79/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3lstm_79_lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_79/lstm_cell_80/MatMulMatMul lstm_79/strided_slice_2:output:02lstm_79/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5lstm_79_lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_79/lstm_cell_80/MatMul_1MatMullstm_79/zeros:output:04lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/addAddV2%lstm_79/lstm_cell_80/MatMul:product:0'lstm_79/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4lstm_79_lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_79/lstm_cell_80/BiasAddBiasAddlstm_79/lstm_cell_80/add:z:03lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_79/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_79/lstm_cell_80/splitSplit-lstm_79/lstm_cell_80/split/split_dim:output:0%lstm_79/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split
lstm_79/lstm_cell_80/SigmoidSigmoid#lstm_79/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/Sigmoid_1Sigmoid#lstm_79/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/mulMul"lstm_79/lstm_cell_80/Sigmoid_1:y:0lstm_79/zeros_1:output:0*
T0*(
_output_shapes
:����������y
lstm_79/lstm_cell_80/ReluRelu#lstm_79/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/mul_1Mul lstm_79/lstm_cell_80/Sigmoid:y:0'lstm_79/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/add_1AddV2lstm_79/lstm_cell_80/mul:z:0lstm_79/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/Sigmoid_2Sigmoid#lstm_79/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������v
lstm_79/lstm_cell_80/Relu_1Relulstm_79/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/mul_2Mul"lstm_79/lstm_cell_80/Sigmoid_2:y:0)lstm_79/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������v
%lstm_79/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   f
$lstm_79/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_79/TensorArrayV2_1TensorListReserve.lstm_79/TensorArrayV2_1/element_shape:output:0-lstm_79/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_79/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_79/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_79/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_79/whileWhile#lstm_79/while/loop_counter:output:0)lstm_79/while/maximum_iterations:output:0lstm_79/time:output:0 lstm_79/TensorArrayV2_1:handle:0lstm_79/zeros:output:0lstm_79/zeros_1:output:0 lstm_79/strided_slice_1:output:0?lstm_79/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_79_lstm_cell_80_matmul_readvariableop_resource5lstm_79_lstm_cell_80_matmul_1_readvariableop_resource4lstm_79_lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_79_while_body_22293024*'
condR
lstm_79_while_cond_22293023*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
8lstm_79/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
*lstm_79/TensorArrayV2Stack/TensorListStackTensorListStacklstm_79/while:output:3Alstm_79/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsp
lstm_79/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_79/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_79/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_slice_3StridedSlice3lstm_79/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_79/strided_slice_3/stack:output:0(lstm_79/strided_slice_3/stack_1:output:0(lstm_79/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskm
lstm_79/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_79/transpose_1	Transpose3lstm_79/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_79/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������c
lstm_79/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_44/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_44/dropout/MulMul lstm_79/strided_slice_3:output:0!dropout_44/dropout/Const:output:0*
T0*(
_output_shapes
:����������h
dropout_44/dropout/ShapeShape lstm_79/strided_slice_3:output:0*
T0*
_output_shapes
:�
/dropout_44/dropout/random_uniform/RandomUniformRandomUniform!dropout_44/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_44/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_44/dropout/GreaterEqualGreaterEqual8dropout_44/dropout/random_uniform/RandomUniform:output:0*dropout_44/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������_
dropout_44/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_44/dropout/SelectV2SelectV2#dropout_44/dropout/GreaterEqual:z:0dropout_44/dropout/Mul:z:0#dropout_44/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_61/MatMulMatMul$dropout_44/dropout/SelectV2:output:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_61/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp,^lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp+^lstm_79/lstm_cell_80/MatMul/ReadVariableOp-^lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp^lstm_79/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2Z
+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp2X
*lstm_79/lstm_cell_80/MatMul/ReadVariableOp*lstm_79/lstm_cell_80/MatMul/ReadVariableOp2\
,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp2
lstm_79/whilelstm_79/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
lstm_79_while_cond_22292871,
(lstm_79_while_lstm_79_while_loop_counter2
.lstm_79_while_lstm_79_while_maximum_iterations
lstm_79_while_placeholder
lstm_79_while_placeholder_1
lstm_79_while_placeholder_2
lstm_79_while_placeholder_3.
*lstm_79_while_less_lstm_79_strided_slice_1F
Blstm_79_while_lstm_79_while_cond_22292871___redundant_placeholder0F
Blstm_79_while_lstm_79_while_cond_22292871___redundant_placeholder1F
Blstm_79_while_lstm_79_while_cond_22292871___redundant_placeholder2F
Blstm_79_while_lstm_79_while_cond_22292871___redundant_placeholder3
lstm_79_while_identity
�
lstm_79/while/LessLesslstm_79_while_placeholder*lstm_79_while_less_lstm_79_strided_slice_1*
T0*
_output_shapes
: [
lstm_79/while/IdentityIdentitylstm_79/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_79_while_identitylstm_79/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
0__inference_sequential_63_layer_call_fn_22292797

inputs
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292454o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)sequential_63_lstm_79_while_cond_22291824H
Dsequential_63_lstm_79_while_sequential_63_lstm_79_while_loop_counterN
Jsequential_63_lstm_79_while_sequential_63_lstm_79_while_maximum_iterations+
'sequential_63_lstm_79_while_placeholder-
)sequential_63_lstm_79_while_placeholder_1-
)sequential_63_lstm_79_while_placeholder_2-
)sequential_63_lstm_79_while_placeholder_3J
Fsequential_63_lstm_79_while_less_sequential_63_lstm_79_strided_slice_1b
^sequential_63_lstm_79_while_sequential_63_lstm_79_while_cond_22291824___redundant_placeholder0b
^sequential_63_lstm_79_while_sequential_63_lstm_79_while_cond_22291824___redundant_placeholder1b
^sequential_63_lstm_79_while_sequential_63_lstm_79_while_cond_22291824___redundant_placeholder2b
^sequential_63_lstm_79_while_sequential_63_lstm_79_while_cond_22291824___redundant_placeholder3(
$sequential_63_lstm_79_while_identity
�
 sequential_63/lstm_79/while/LessLess'sequential_63_lstm_79_while_placeholderFsequential_63_lstm_79_while_less_sequential_63_lstm_79_strided_slice_1*
T0*
_output_shapes
: w
$sequential_63/lstm_79/while/IdentityIdentity$sequential_63/lstm_79/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_63_lstm_79_while_identity-sequential_63/lstm_79/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22291984

inputs

states
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������O
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�
�
0__inference_sequential_63_layer_call_fn_22292812

inputs
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292701o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_22292573
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_80_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_80_matmul_readvariableop_resource:	�G
3while_lstm_cell_80_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_80_biasadd_readvariableop_resource:	���)while/lstm_cell_80/BiasAdd/ReadVariableOp�(while/lstm_cell_80/MatMul/ReadVariableOp�*while/lstm_cell_80/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_80/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_80/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/addAddV2#while/lstm_cell_80/MatMul:product:0%while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_80/BiasAddBiasAddwhile/lstm_cell_80/add:z:01while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_80/splitSplit+while/lstm_cell_80/split/split_dim:output:0#while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_80/SigmoidSigmoid!while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_1Sigmoid!while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mulMul while/lstm_cell_80/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_80/ReluRelu!while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_1Mulwhile/lstm_cell_80/Sigmoid:y:0%while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/add_1AddV2while/lstm_cell_80/mul:z:0while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_2Sigmoid!while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_80/Relu_1Reluwhile/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_2Mul while/lstm_cell_80/Sigmoid_2:y:0'while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_80/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_80/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_80/BiasAdd/ReadVariableOp)^while/lstm_cell_80/MatMul/ReadVariableOp+^while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_80_biasadd_readvariableop_resource4while_lstm_cell_80_biasadd_readvariableop_resource_0"l
3while_lstm_cell_80_matmul_1_readvariableop_resource5while_lstm_cell_80_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_80_matmul_readvariableop_resource3while_lstm_cell_80_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_80/BiasAdd/ReadVariableOp)while/lstm_cell_80/BiasAdd/ReadVariableOp2T
(while/lstm_cell_80/MatMul/ReadVariableOp(while/lstm_cell_80/MatMul/ReadVariableOp2X
*while/lstm_cell_80/MatMul_1/ReadVariableOp*while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_22293661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22293661___redundant_placeholder06
2while_while_cond_22293661___redundant_placeholder16
2while_while_cond_22293661___redundant_placeholder26
2while_while_cond_22293661___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292746
lstm_79_input#
lstm_79_22292732:	�$
lstm_79_22292734:
��
lstm_79_22292736:	�$
dense_61_22292740:	�
dense_61_22292742:
identity�� dense_61/StatefulPartitionedCall�lstm_79/StatefulPartitionedCall�
lstm_79/StatefulPartitionedCallStatefulPartitionedCalllstm_79_inputlstm_79_22292732lstm_79_22292734lstm_79_22292736*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292422�
dropout_44/PartitionedCallPartitionedCall(lstm_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292435�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0dense_61_22292740dense_61_22292742*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_22292447x
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_61/StatefulPartitionedCall ^lstm_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2B
lstm_79/StatefulPartitionedCalllstm_79/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_79_input
�
�
while_cond_22293226
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22293226___redundant_placeholder06
2while_while_cond_22293226___redundant_placeholder16
2while_while_cond_22293226___redundant_placeholder26
2while_while_cond_22293226___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_22293227
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_80_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_80_matmul_readvariableop_resource:	�G
3while_lstm_cell_80_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_80_biasadd_readvariableop_resource:	���)while/lstm_cell_80/BiasAdd/ReadVariableOp�(while/lstm_cell_80/MatMul/ReadVariableOp�*while/lstm_cell_80/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_80/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_80/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/addAddV2#while/lstm_cell_80/MatMul:product:0%while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_80/BiasAddBiasAddwhile/lstm_cell_80/add:z:01while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_80/splitSplit+while/lstm_cell_80/split/split_dim:output:0#while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_80/SigmoidSigmoid!while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_1Sigmoid!while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mulMul while/lstm_cell_80/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_80/ReluRelu!while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_1Mulwhile/lstm_cell_80/Sigmoid:y:0%while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/add_1AddV2while/lstm_cell_80/mul:z:0while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_2Sigmoid!while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_80/Relu_1Reluwhile/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_2Mul while/lstm_cell_80/Sigmoid_2:y:0'while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_80/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_80/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_80/BiasAdd/ReadVariableOp)^while/lstm_cell_80/MatMul/ReadVariableOp+^while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_80_biasadd_readvariableop_resource4while_lstm_cell_80_biasadd_readvariableop_resource_0"l
3while_lstm_cell_80_matmul_1_readvariableop_resource5while_lstm_cell_80_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_80_matmul_readvariableop_resource3while_lstm_cell_80_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_80/BiasAdd/ReadVariableOp)while/lstm_cell_80/BiasAdd/ReadVariableOp2T
(while/lstm_cell_80/MatMul/ReadVariableOp(while/lstm_cell_80/MatMul/ReadVariableOp2X
*while/lstm_cell_80/MatMul_1/ReadVariableOp*while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292763
lstm_79_input#
lstm_79_22292749:	�$
lstm_79_22292751:
��
lstm_79_22292753:	�$
dense_61_22292757:	�
dense_61_22292759:
identity�� dense_61/StatefulPartitionedCall�"dropout_44/StatefulPartitionedCall�lstm_79/StatefulPartitionedCall�
lstm_79/StatefulPartitionedCallStatefulPartitionedCalllstm_79_inputlstm_79_22292749lstm_79_22292751lstm_79_22292753*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292658�
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall(lstm_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292497�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0dense_61_22292757dense_61_22292759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_22292447x
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_61/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall ^lstm_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall2B
lstm_79/StatefulPartitionedCalllstm_79/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_79_input
�
�
0__inference_sequential_63_layer_call_fn_22292467
lstm_79_input
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_79_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292454o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_79_input
�
�
*__inference_lstm_79_layer_call_fn_22293167

inputs
unknown:	�
	unknown_0:
��
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292658p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
while_body_22292192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_80_22292216_0:	�1
while_lstm_cell_80_22292218_0:
��,
while_lstm_cell_80_22292220_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_80_22292216:	�/
while_lstm_cell_80_22292218:
��*
while_lstm_cell_80_22292220:	���*while/lstm_cell_80/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_80/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_80_22292216_0while_lstm_cell_80_22292218_0while_lstm_cell_80_22292220_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22292132r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_80/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity3while/lstm_cell_80/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:�����������
while/Identity_5Identity3while/lstm_cell_80/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:����������y

while/NoOpNoOp+^while/lstm_cell_80/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_80_22292216while_lstm_cell_80_22292216_0"<
while_lstm_cell_80_22292218while_lstm_cell_80_22292218_0"<
while_lstm_cell_80_22292220while_lstm_cell_80_22292220_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2X
*while/lstm_cell_80/StatefulPartitionedCall*while/lstm_cell_80/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�	
�
F__inference_dense_61_layer_call_and_return_conditional_losses_22292447

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�]
�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292964

inputsF
3lstm_79_lstm_cell_80_matmul_readvariableop_resource:	�I
5lstm_79_lstm_cell_80_matmul_1_readvariableop_resource:
��C
4lstm_79_lstm_cell_80_biasadd_readvariableop_resource:	�:
'dense_61_matmul_readvariableop_resource:	�6
(dense_61_biasadd_readvariableop_resource:
identity��dense_61/BiasAdd/ReadVariableOp�dense_61/MatMul/ReadVariableOp�+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp�*lstm_79/lstm_cell_80/MatMul/ReadVariableOp�,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp�lstm_79/whileC
lstm_79/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_79/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_79/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_79/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_sliceStridedSlicelstm_79/Shape:output:0$lstm_79/strided_slice/stack:output:0&lstm_79/strided_slice/stack_1:output:0&lstm_79/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_79/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_79/zeros/packedPacklstm_79/strided_slice:output:0lstm_79/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_79/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_79/zerosFilllstm_79/zeros/packed:output:0lstm_79/zeros/Const:output:0*
T0*(
_output_shapes
:����������[
lstm_79/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_79/zeros_1/packedPacklstm_79/strided_slice:output:0!lstm_79/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_79/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_79/zeros_1Filllstm_79/zeros_1/packed:output:0lstm_79/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������k
lstm_79/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_79/transpose	Transposeinputslstm_79/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_79/Shape_1Shapelstm_79/transpose:y:0*
T0*
_output_shapes
:g
lstm_79/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_79/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_79/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_slice_1StridedSlicelstm_79/Shape_1:output:0&lstm_79/strided_slice_1/stack:output:0(lstm_79/strided_slice_1/stack_1:output:0(lstm_79/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_79/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_79/TensorArrayV2TensorListReserve,lstm_79/TensorArrayV2/element_shape:output:0 lstm_79/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_79/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_79/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_79/transpose:y:0Flstm_79/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_79/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_79/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_79/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_slice_2StridedSlicelstm_79/transpose:y:0&lstm_79/strided_slice_2/stack:output:0(lstm_79/strided_slice_2/stack_1:output:0(lstm_79/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_79/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3lstm_79_lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_79/lstm_cell_80/MatMulMatMul lstm_79/strided_slice_2:output:02lstm_79/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5lstm_79_lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_79/lstm_cell_80/MatMul_1MatMullstm_79/zeros:output:04lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/addAddV2%lstm_79/lstm_cell_80/MatMul:product:0'lstm_79/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4lstm_79_lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_79/lstm_cell_80/BiasAddBiasAddlstm_79/lstm_cell_80/add:z:03lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_79/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_79/lstm_cell_80/splitSplit-lstm_79/lstm_cell_80/split/split_dim:output:0%lstm_79/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split
lstm_79/lstm_cell_80/SigmoidSigmoid#lstm_79/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/Sigmoid_1Sigmoid#lstm_79/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/mulMul"lstm_79/lstm_cell_80/Sigmoid_1:y:0lstm_79/zeros_1:output:0*
T0*(
_output_shapes
:����������y
lstm_79/lstm_cell_80/ReluRelu#lstm_79/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/mul_1Mul lstm_79/lstm_cell_80/Sigmoid:y:0'lstm_79/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/add_1AddV2lstm_79/lstm_cell_80/mul:z:0lstm_79/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/Sigmoid_2Sigmoid#lstm_79/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������v
lstm_79/lstm_cell_80/Relu_1Relulstm_79/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_79/lstm_cell_80/mul_2Mul"lstm_79/lstm_cell_80/Sigmoid_2:y:0)lstm_79/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������v
%lstm_79/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   f
$lstm_79/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_79/TensorArrayV2_1TensorListReserve.lstm_79/TensorArrayV2_1/element_shape:output:0-lstm_79/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_79/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_79/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_79/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_79/whileWhile#lstm_79/while/loop_counter:output:0)lstm_79/while/maximum_iterations:output:0lstm_79/time:output:0 lstm_79/TensorArrayV2_1:handle:0lstm_79/zeros:output:0lstm_79/zeros_1:output:0 lstm_79/strided_slice_1:output:0?lstm_79/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_79_lstm_cell_80_matmul_readvariableop_resource5lstm_79_lstm_cell_80_matmul_1_readvariableop_resource4lstm_79_lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_79_while_body_22292872*'
condR
lstm_79_while_cond_22292871*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
8lstm_79/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
*lstm_79/TensorArrayV2Stack/TensorListStackTensorListStacklstm_79/while:output:3Alstm_79/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsp
lstm_79/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_79/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_79/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_79/strided_slice_3StridedSlice3lstm_79/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_79/strided_slice_3/stack:output:0(lstm_79/strided_slice_3/stack_1:output:0(lstm_79/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskm
lstm_79/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_79/transpose_1	Transpose3lstm_79/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_79/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������c
lstm_79/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
dropout_44/IdentityIdentity lstm_79/strided_slice_3:output:0*
T0*(
_output_shapes
:�����������
dense_61/MatMul/ReadVariableOpReadVariableOp'dense_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_61/MatMulMatMuldropout_44/Identity:output:0&dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_61/BiasAdd/ReadVariableOpReadVariableOp(dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_61/BiasAddBiasAdddense_61/MatMul:product:0'dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_61/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_61/BiasAdd/ReadVariableOp^dense_61/MatMul/ReadVariableOp,^lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp+^lstm_79/lstm_cell_80/MatMul/ReadVariableOp-^lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp^lstm_79/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_61/BiasAdd/ReadVariableOpdense_61/BiasAdd/ReadVariableOp2@
dense_61/MatMul/ReadVariableOpdense_61/MatMul/ReadVariableOp2Z
+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp+lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp2X
*lstm_79/lstm_cell_80/MatMul/ReadVariableOp*lstm_79/lstm_cell_80/MatMul/ReadVariableOp2\
,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp,lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp2
lstm_79/whilelstm_79/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_22293372
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_80_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_80_matmul_readvariableop_resource:	�G
3while_lstm_cell_80_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_80_biasadd_readvariableop_resource:	���)while/lstm_cell_80/BiasAdd/ReadVariableOp�(while/lstm_cell_80/MatMul/ReadVariableOp�*while/lstm_cell_80/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_80/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_80/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/addAddV2#while/lstm_cell_80/MatMul:product:0%while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_80/BiasAddBiasAddwhile/lstm_cell_80/add:z:01while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_80/splitSplit+while/lstm_cell_80/split/split_dim:output:0#while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_80/SigmoidSigmoid!while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_1Sigmoid!while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mulMul while/lstm_cell_80/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_80/ReluRelu!while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_1Mulwhile/lstm_cell_80/Sigmoid:y:0%while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/add_1AddV2while/lstm_cell_80/mul:z:0while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_2Sigmoid!while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_80/Relu_1Reluwhile/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_2Mul while/lstm_cell_80/Sigmoid_2:y:0'while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_80/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_80/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_80/BiasAdd/ReadVariableOp)^while/lstm_cell_80/MatMul/ReadVariableOp+^while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_80_biasadd_readvariableop_resource4while_lstm_cell_80_biasadd_readvariableop_resource_0"l
3while_lstm_cell_80_matmul_1_readvariableop_resource5while_lstm_cell_80_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_80_matmul_readvariableop_resource3while_lstm_cell_80_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_80/BiasAdd/ReadVariableOp)while/lstm_cell_80/BiasAdd/ReadVariableOp2T
(while/lstm_cell_80/MatMul/ReadVariableOp(while/lstm_cell_80/MatMul/ReadVariableOp2X
*while/lstm_cell_80/MatMul_1/ReadVariableOp*while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_dense_61_layer_call_fn_22293783

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_22292447o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_dropout_44_layer_call_fn_22293752

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292435a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�K
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293457
inputs_0>
+lstm_cell_80_matmul_readvariableop_resource:	�A
-lstm_cell_80_matmul_1_readvariableop_resource:
��;
,lstm_cell_80_biasadd_readvariableop_resource:	�
identity��#lstm_cell_80/BiasAdd/ReadVariableOp�"lstm_cell_80/MatMul/ReadVariableOp�$lstm_cell_80/MatMul_1/ReadVariableOp�while=
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_80/MatMul/ReadVariableOpReadVariableOp+lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_80/MatMulMatMulstrided_slice_2:output:0*lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_80/MatMul_1MatMulzeros:output:0,lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/addAddV2lstm_cell_80/MatMul:product:0lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_80/BiasAddBiasAddlstm_cell_80/add:z:0+lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_80/splitSplit%lstm_cell_80/split/split_dim:output:0lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_80/SigmoidSigmoidlstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_1Sigmoidlstm_cell_80/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_80/mulMullstm_cell_80/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_80/ReluRelulstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_1Mullstm_cell_80/Sigmoid:y:0lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_80/add_1AddV2lstm_cell_80/mul:z:0lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_2Sigmoidlstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_80/Relu_1Relulstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_2Mullstm_cell_80/Sigmoid_2:y:0!lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_80_matmul_readvariableop_resource-lstm_cell_80_matmul_1_readvariableop_resource,lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22293372*
condR
while_cond_22293371*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp$^lstm_cell_80/BiasAdd/ReadVariableOp#^lstm_cell_80/MatMul/ReadVariableOp%^lstm_cell_80/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_80/BiasAdd/ReadVariableOp#lstm_cell_80/BiasAdd/ReadVariableOp2H
"lstm_cell_80/MatMul/ReadVariableOp"lstm_cell_80/MatMul/ReadVariableOp2L
$lstm_cell_80/MatMul_1/ReadVariableOp$lstm_cell_80/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292701

inputs#
lstm_79_22292687:	�$
lstm_79_22292689:
��
lstm_79_22292691:	�$
dense_61_22292695:	�
dense_61_22292697:
identity�� dense_61/StatefulPartitionedCall�"dropout_44/StatefulPartitionedCall�lstm_79/StatefulPartitionedCall�
lstm_79/StatefulPartitionedCallStatefulPartitionedCallinputslstm_79_22292687lstm_79_22292689lstm_79_22292691*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292658�
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall(lstm_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292497�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0dense_61_22292695dense_61_22292697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_22292447x
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_61/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall ^lstm_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall2B
lstm_79/StatefulPartitionedCalllstm_79/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293762

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�C
�

lstm_79_while_body_22292872,
(lstm_79_while_lstm_79_while_loop_counter2
.lstm_79_while_lstm_79_while_maximum_iterations
lstm_79_while_placeholder
lstm_79_while_placeholder_1
lstm_79_while_placeholder_2
lstm_79_while_placeholder_3+
'lstm_79_while_lstm_79_strided_slice_1_0g
clstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0:	�Q
=lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��K
<lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
lstm_79_while_identity
lstm_79_while_identity_1
lstm_79_while_identity_2
lstm_79_while_identity_3
lstm_79_while_identity_4
lstm_79_while_identity_5)
%lstm_79_while_lstm_79_strided_slice_1e
alstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensorL
9lstm_79_while_lstm_cell_80_matmul_readvariableop_resource:	�O
;lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource:
��I
:lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource:	���1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp�0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp�2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp�
?lstm_79/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_79/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensor_0lstm_79_while_placeholderHlstm_79/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp;lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_79/while/lstm_cell_80/MatMulMatMul8lstm_79/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp=lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
#lstm_79/while/lstm_cell_80/MatMul_1MatMullstm_79_while_placeholder_2:lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_79/while/lstm_cell_80/addAddV2+lstm_79/while/lstm_cell_80/MatMul:product:0-lstm_79/while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp<lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_79/while/lstm_cell_80/BiasAddBiasAdd"lstm_79/while/lstm_cell_80/add:z:09lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_79/while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_79/while/lstm_cell_80/splitSplit3lstm_79/while/lstm_cell_80/split/split_dim:output:0+lstm_79/while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
"lstm_79/while/lstm_cell_80/SigmoidSigmoid)lstm_79/while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:�����������
$lstm_79/while/lstm_cell_80/Sigmoid_1Sigmoid)lstm_79/while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
lstm_79/while/lstm_cell_80/mulMul(lstm_79/while/lstm_cell_80/Sigmoid_1:y:0lstm_79_while_placeholder_3*
T0*(
_output_shapes
:�����������
lstm_79/while/lstm_cell_80/ReluRelu)lstm_79/while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
 lstm_79/while/lstm_cell_80/mul_1Mul&lstm_79/while/lstm_cell_80/Sigmoid:y:0-lstm_79/while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
 lstm_79/while/lstm_cell_80/add_1AddV2"lstm_79/while/lstm_cell_80/mul:z:0$lstm_79/while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:�����������
$lstm_79/while/lstm_cell_80/Sigmoid_2Sigmoid)lstm_79/while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:�����������
!lstm_79/while/lstm_cell_80/Relu_1Relu$lstm_79/while/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
 lstm_79/while/lstm_cell_80/mul_2Mul(lstm_79/while/lstm_cell_80/Sigmoid_2:y:0/lstm_79/while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������z
8lstm_79/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_79/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_79_while_placeholder_1Alstm_79/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_79/while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_79/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_79/while/addAddV2lstm_79_while_placeholderlstm_79/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_79/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_79/while/add_1AddV2(lstm_79_while_lstm_79_while_loop_counterlstm_79/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_79/while/IdentityIdentitylstm_79/while/add_1:z:0^lstm_79/while/NoOp*
T0*
_output_shapes
: �
lstm_79/while/Identity_1Identity.lstm_79_while_lstm_79_while_maximum_iterations^lstm_79/while/NoOp*
T0*
_output_shapes
: q
lstm_79/while/Identity_2Identitylstm_79/while/add:z:0^lstm_79/while/NoOp*
T0*
_output_shapes
: �
lstm_79/while/Identity_3IdentityBlstm_79/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_79/while/NoOp*
T0*
_output_shapes
: �
lstm_79/while/Identity_4Identity$lstm_79/while/lstm_cell_80/mul_2:z:0^lstm_79/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_79/while/Identity_5Identity$lstm_79/while/lstm_cell_80/add_1:z:0^lstm_79/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_79/while/NoOpNoOp2^lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp1^lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp3^lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_79_while_identitylstm_79/while/Identity:output:0"=
lstm_79_while_identity_1!lstm_79/while/Identity_1:output:0"=
lstm_79_while_identity_2!lstm_79/while/Identity_2:output:0"=
lstm_79_while_identity_3!lstm_79/while/Identity_3:output:0"=
lstm_79_while_identity_4!lstm_79/while/Identity_4:output:0"=
lstm_79_while_identity_5!lstm_79/while/Identity_5:output:0"P
%lstm_79_while_lstm_79_strided_slice_1'lstm_79_while_lstm_79_strided_slice_1_0"z
:lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource<lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0"|
;lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource=lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0"x
9lstm_79_while_lstm_cell_80_matmul_readvariableop_resource;lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0"�
alstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensorclstm_79_while_tensorarrayv2read_tensorlistgetitem_lstm_79_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2f
1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp1lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp2d
0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp0lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp2h
2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp2lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293891

inputs
states_0
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������O
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states_0:RN
(
_output_shapes
:����������
"
_user_specified_name
states_1
�

g
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292497

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292454

inputs#
lstm_79_22292423:	�$
lstm_79_22292425:
��
lstm_79_22292427:	�$
dense_61_22292448:	�
dense_61_22292450:
identity�� dense_61/StatefulPartitionedCall�lstm_79/StatefulPartitionedCall�
lstm_79/StatefulPartitionedCallStatefulPartitionedCallinputslstm_79_22292423lstm_79_22292425lstm_79_22292427*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292422�
dropout_44/PartitionedCallPartitionedCall(lstm_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292435�
 dense_61/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0dense_61_22292448dense_61_22292450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_61_layer_call_and_return_conditional_losses_22292447x
IdentityIdentity)dense_61/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_61/StatefulPartitionedCall ^lstm_79/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_61/StatefulPartitionedCall dense_61/StatefulPartitionedCall2B
lstm_79/StatefulPartitionedCalllstm_79/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22292132

inputs

states
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������O
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�
�
&__inference_signature_wrapper_22292782
lstm_79_input
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_79_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_22291917o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_79_input
�

�
lstm_79_while_cond_22293023,
(lstm_79_while_lstm_79_while_loop_counter2
.lstm_79_while_lstm_79_while_maximum_iterations
lstm_79_while_placeholder
lstm_79_while_placeholder_1
lstm_79_while_placeholder_2
lstm_79_while_placeholder_3.
*lstm_79_while_less_lstm_79_strided_slice_1F
Blstm_79_while_lstm_79_while_cond_22293023___redundant_placeholder0F
Blstm_79_while_lstm_79_while_cond_22293023___redundant_placeholder1F
Blstm_79_while_lstm_79_while_cond_22293023___redundant_placeholder2F
Blstm_79_while_lstm_79_while_cond_22293023___redundant_placeholder3
lstm_79_while_identity
�
lstm_79/while/LessLesslstm_79_while_placeholder*lstm_79_while_less_lstm_79_strided_slice_1*
T0*
_output_shapes
: [
lstm_79/while/IdentityIdentitylstm_79/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_79_while_identitylstm_79/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�

g
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293774

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
-__inference_dropout_44_layer_call_fn_22293757

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_44_layer_call_and_return_conditional_losses_22292497p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�S
�
)sequential_63_lstm_79_while_body_22291825H
Dsequential_63_lstm_79_while_sequential_63_lstm_79_while_loop_counterN
Jsequential_63_lstm_79_while_sequential_63_lstm_79_while_maximum_iterations+
'sequential_63_lstm_79_while_placeholder-
)sequential_63_lstm_79_while_placeholder_1-
)sequential_63_lstm_79_while_placeholder_2-
)sequential_63_lstm_79_while_placeholder_3G
Csequential_63_lstm_79_while_sequential_63_lstm_79_strided_slice_1_0�
sequential_63_lstm_79_while_tensorarrayv2read_tensorlistgetitem_sequential_63_lstm_79_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_63_lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0:	�_
Ksequential_63_lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��Y
Jsequential_63_lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0:	�(
$sequential_63_lstm_79_while_identity*
&sequential_63_lstm_79_while_identity_1*
&sequential_63_lstm_79_while_identity_2*
&sequential_63_lstm_79_while_identity_3*
&sequential_63_lstm_79_while_identity_4*
&sequential_63_lstm_79_while_identity_5E
Asequential_63_lstm_79_while_sequential_63_lstm_79_strided_slice_1�
}sequential_63_lstm_79_while_tensorarrayv2read_tensorlistgetitem_sequential_63_lstm_79_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_63_lstm_79_while_lstm_cell_80_matmul_readvariableop_resource:	�]
Isequential_63_lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource:
��W
Hsequential_63_lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource:	���?sequential_63/lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp�>sequential_63/lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp�@sequential_63/lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp�
Msequential_63/lstm_79/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
?sequential_63/lstm_79/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_63_lstm_79_while_tensorarrayv2read_tensorlistgetitem_sequential_63_lstm_79_tensorarrayunstack_tensorlistfromtensor_0'sequential_63_lstm_79_while_placeholderVsequential_63/lstm_79/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
>sequential_63/lstm_79/while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOpIsequential_63_lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
/sequential_63/lstm_79/while/lstm_cell_80/MatMulMatMulFsequential_63/lstm_79/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_63/lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@sequential_63/lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOpKsequential_63_lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
1sequential_63/lstm_79/while/lstm_cell_80/MatMul_1MatMul)sequential_63_lstm_79_while_placeholder_2Hsequential_63/lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_63/lstm_79/while/lstm_cell_80/addAddV29sequential_63/lstm_79/while/lstm_cell_80/MatMul:product:0;sequential_63/lstm_79/while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
?sequential_63/lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOpJsequential_63_lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
0sequential_63/lstm_79/while/lstm_cell_80/BiasAddBiasAdd0sequential_63/lstm_79/while/lstm_cell_80/add:z:0Gsequential_63/lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
8sequential_63/lstm_79/while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
.sequential_63/lstm_79/while/lstm_cell_80/splitSplitAsequential_63/lstm_79/while/lstm_cell_80/split/split_dim:output:09sequential_63/lstm_79/while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
0sequential_63/lstm_79/while/lstm_cell_80/SigmoidSigmoid7sequential_63/lstm_79/while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:�����������
2sequential_63/lstm_79/while/lstm_cell_80/Sigmoid_1Sigmoid7sequential_63/lstm_79/while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
,sequential_63/lstm_79/while/lstm_cell_80/mulMul6sequential_63/lstm_79/while/lstm_cell_80/Sigmoid_1:y:0)sequential_63_lstm_79_while_placeholder_3*
T0*(
_output_shapes
:�����������
-sequential_63/lstm_79/while/lstm_cell_80/ReluRelu7sequential_63/lstm_79/while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
.sequential_63/lstm_79/while/lstm_cell_80/mul_1Mul4sequential_63/lstm_79/while/lstm_cell_80/Sigmoid:y:0;sequential_63/lstm_79/while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
.sequential_63/lstm_79/while/lstm_cell_80/add_1AddV20sequential_63/lstm_79/while/lstm_cell_80/mul:z:02sequential_63/lstm_79/while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:�����������
2sequential_63/lstm_79/while/lstm_cell_80/Sigmoid_2Sigmoid7sequential_63/lstm_79/while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:�����������
/sequential_63/lstm_79/while/lstm_cell_80/Relu_1Relu2sequential_63/lstm_79/while/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
.sequential_63/lstm_79/while/lstm_cell_80/mul_2Mul6sequential_63/lstm_79/while/lstm_cell_80/Sigmoid_2:y:0=sequential_63/lstm_79/while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:�����������
Fsequential_63/lstm_79/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
@sequential_63/lstm_79/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_63_lstm_79_while_placeholder_1Osequential_63/lstm_79/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_63/lstm_79/while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���c
!sequential_63/lstm_79/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_63/lstm_79/while/addAddV2'sequential_63_lstm_79_while_placeholder*sequential_63/lstm_79/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_63/lstm_79/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
!sequential_63/lstm_79/while/add_1AddV2Dsequential_63_lstm_79_while_sequential_63_lstm_79_while_loop_counter,sequential_63/lstm_79/while/add_1/y:output:0*
T0*
_output_shapes
: �
$sequential_63/lstm_79/while/IdentityIdentity%sequential_63/lstm_79/while/add_1:z:0!^sequential_63/lstm_79/while/NoOp*
T0*
_output_shapes
: �
&sequential_63/lstm_79/while/Identity_1IdentityJsequential_63_lstm_79_while_sequential_63_lstm_79_while_maximum_iterations!^sequential_63/lstm_79/while/NoOp*
T0*
_output_shapes
: �
&sequential_63/lstm_79/while/Identity_2Identity#sequential_63/lstm_79/while/add:z:0!^sequential_63/lstm_79/while/NoOp*
T0*
_output_shapes
: �
&sequential_63/lstm_79/while/Identity_3IdentityPsequential_63/lstm_79/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_63/lstm_79/while/NoOp*
T0*
_output_shapes
: �
&sequential_63/lstm_79/while/Identity_4Identity2sequential_63/lstm_79/while/lstm_cell_80/mul_2:z:0!^sequential_63/lstm_79/while/NoOp*
T0*(
_output_shapes
:�����������
&sequential_63/lstm_79/while/Identity_5Identity2sequential_63/lstm_79/while/lstm_cell_80/add_1:z:0!^sequential_63/lstm_79/while/NoOp*
T0*(
_output_shapes
:�����������
 sequential_63/lstm_79/while/NoOpNoOp@^sequential_63/lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp?^sequential_63/lstm_79/while/lstm_cell_80/MatMul/ReadVariableOpA^sequential_63/lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_63_lstm_79_while_identity-sequential_63/lstm_79/while/Identity:output:0"Y
&sequential_63_lstm_79_while_identity_1/sequential_63/lstm_79/while/Identity_1:output:0"Y
&sequential_63_lstm_79_while_identity_2/sequential_63/lstm_79/while/Identity_2:output:0"Y
&sequential_63_lstm_79_while_identity_3/sequential_63/lstm_79/while/Identity_3:output:0"Y
&sequential_63_lstm_79_while_identity_4/sequential_63/lstm_79/while/Identity_4:output:0"Y
&sequential_63_lstm_79_while_identity_5/sequential_63/lstm_79/while/Identity_5:output:0"�
Hsequential_63_lstm_79_while_lstm_cell_80_biasadd_readvariableop_resourceJsequential_63_lstm_79_while_lstm_cell_80_biasadd_readvariableop_resource_0"�
Isequential_63_lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resourceKsequential_63_lstm_79_while_lstm_cell_80_matmul_1_readvariableop_resource_0"�
Gsequential_63_lstm_79_while_lstm_cell_80_matmul_readvariableop_resourceIsequential_63_lstm_79_while_lstm_cell_80_matmul_readvariableop_resource_0"�
Asequential_63_lstm_79_while_sequential_63_lstm_79_strided_slice_1Csequential_63_lstm_79_while_sequential_63_lstm_79_strided_slice_1_0"�
}sequential_63_lstm_79_while_tensorarrayv2read_tensorlistgetitem_sequential_63_lstm_79_tensorarrayunstack_tensorlistfromtensorsequential_63_lstm_79_while_tensorarrayv2read_tensorlistgetitem_sequential_63_lstm_79_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2�
?sequential_63/lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp?sequential_63/lstm_79/while/lstm_cell_80/BiasAdd/ReadVariableOp2�
>sequential_63/lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp>sequential_63/lstm_79/while/lstm_cell_80/MatMul/ReadVariableOp2�
@sequential_63/lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp@sequential_63/lstm_79/while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�K
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292422

inputs>
+lstm_cell_80_matmul_readvariableop_resource:	�A
-lstm_cell_80_matmul_1_readvariableop_resource:
��;
,lstm_cell_80_biasadd_readvariableop_resource:	�
identity��#lstm_cell_80/BiasAdd/ReadVariableOp�"lstm_cell_80/MatMul/ReadVariableOp�$lstm_cell_80/MatMul_1/ReadVariableOp�while;
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_80/MatMul/ReadVariableOpReadVariableOp+lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_80/MatMulMatMulstrided_slice_2:output:0*lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_80/MatMul_1MatMulzeros:output:0,lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/addAddV2lstm_cell_80/MatMul:product:0lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_80/BiasAddBiasAddlstm_cell_80/add:z:0+lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_80/splitSplit%lstm_cell_80/split/split_dim:output:0lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_80/SigmoidSigmoidlstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_1Sigmoidlstm_cell_80/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_80/mulMullstm_cell_80/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_80/ReluRelulstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_1Mullstm_cell_80/Sigmoid:y:0lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_80/add_1AddV2lstm_cell_80/mul:z:0lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_2Sigmoidlstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_80/Relu_1Relulstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_2Mullstm_cell_80/Sigmoid_2:y:0!lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_80_matmul_readvariableop_resource-lstm_cell_80_matmul_1_readvariableop_resource,lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22292337*
condR
while_cond_22292336*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp$^lstm_cell_80/BiasAdd/ReadVariableOp#^lstm_cell_80/MatMul/ReadVariableOp%^lstm_cell_80/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_80/BiasAdd/ReadVariableOp#lstm_cell_80/BiasAdd/ReadVariableOp2H
"lstm_cell_80/MatMul/ReadVariableOp"lstm_cell_80/MatMul/ReadVariableOp2L
$lstm_cell_80/MatMul_1/ReadVariableOp$lstm_cell_80/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_22292336
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22292336___redundant_placeholder06
2while_while_cond_22292336___redundant_placeholder16
2while_while_cond_22292336___redundant_placeholder26
2while_while_cond_22292336___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_22293517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_80_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_80_matmul_readvariableop_resource:	�G
3while_lstm_cell_80_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_80_biasadd_readvariableop_resource:	���)while/lstm_cell_80/BiasAdd/ReadVariableOp�(while/lstm_cell_80/MatMul/ReadVariableOp�*while/lstm_cell_80/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_80/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_80/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/addAddV2#while/lstm_cell_80/MatMul:product:0%while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_80/BiasAddBiasAddwhile/lstm_cell_80/add:z:01while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_80/splitSplit+while/lstm_cell_80/split/split_dim:output:0#while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_80/SigmoidSigmoid!while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_1Sigmoid!while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mulMul while/lstm_cell_80/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_80/ReluRelu!while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_1Mulwhile/lstm_cell_80/Sigmoid:y:0%while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/add_1AddV2while/lstm_cell_80/mul:z:0while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_2Sigmoid!while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_80/Relu_1Reluwhile/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_2Mul while/lstm_cell_80/Sigmoid_2:y:0'while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_80/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_80/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_80/BiasAdd/ReadVariableOp)^while/lstm_cell_80/MatMul/ReadVariableOp+^while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_80_biasadd_readvariableop_resource4while_lstm_cell_80_biasadd_readvariableop_resource_0"l
3while_lstm_cell_80_matmul_1_readvariableop_resource5while_lstm_cell_80_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_80_matmul_readvariableop_resource3while_lstm_cell_80_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_80/BiasAdd/ReadVariableOp)while/lstm_cell_80/BiasAdd/ReadVariableOp2T
(while/lstm_cell_80/MatMul/ReadVariableOp(while/lstm_cell_80/MatMul/ReadVariableOp2X
*while/lstm_cell_80/MatMul_1/ReadVariableOp*while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_sequential_63_layer_call_fn_22292729
lstm_79_input
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_79_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292701o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_79_input
�
�
*__inference_lstm_79_layer_call_fn_22293156

inputs
unknown:	�
	unknown_0:
��
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292422p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293859

inputs
states_0
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:����������O
ReluRelusplit:output:2*
T0*(
_output_shapes
:����������`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:����������U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:����������d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:����������Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:����������[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states_0:RN
(
_output_shapes
:����������
"
_user_specified_name
states_1
�
�
/__inference_lstm_cell_80_layer_call_fn_22293810

inputs
states_0
states_1
unknown:	�
	unknown_0:
��
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22291984p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:���������:����������:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states_0:RN
(
_output_shapes
:����������
"
_user_specified_name
states_1
�9
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292262

inputs(
lstm_cell_80_22292178:	�)
lstm_cell_80_22292180:
��$
lstm_cell_80_22292182:	�
identity��$lstm_cell_80/StatefulPartitionedCall�while;
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
$lstm_cell_80/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_80_22292178lstm_cell_80_22292180lstm_cell_80_22292182*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22292132n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_80_22292178lstm_cell_80_22292180lstm_cell_80_22292182*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22292192*
condR
while_cond_22292191*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:����������u
NoOpNoOp%^lstm_cell_80/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_80/StatefulPartitionedCall$lstm_cell_80/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�9
�
while_body_22293662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_80_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_80_matmul_readvariableop_resource:	�G
3while_lstm_cell_80_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_80_biasadd_readvariableop_resource:	���)while/lstm_cell_80/BiasAdd/ReadVariableOp�(while/lstm_cell_80/MatMul/ReadVariableOp�*while/lstm_cell_80/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_80/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_80/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/addAddV2#while/lstm_cell_80/MatMul:product:0%while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_80/BiasAddBiasAddwhile/lstm_cell_80/add:z:01while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_80/splitSplit+while/lstm_cell_80/split/split_dim:output:0#while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_80/SigmoidSigmoid!while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_1Sigmoid!while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mulMul while/lstm_cell_80/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_80/ReluRelu!while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_1Mulwhile/lstm_cell_80/Sigmoid:y:0%while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/add_1AddV2while/lstm_cell_80/mul:z:0while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_2Sigmoid!while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_80/Relu_1Reluwhile/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_2Mul while/lstm_cell_80/Sigmoid_2:y:0'while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_80/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_80/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_80/BiasAdd/ReadVariableOp)^while/lstm_cell_80/MatMul/ReadVariableOp+^while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_80_biasadd_readvariableop_resource4while_lstm_cell_80_biasadd_readvariableop_resource_0"l
3while_lstm_cell_80_matmul_1_readvariableop_resource5while_lstm_cell_80_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_80_matmul_readvariableop_resource3while_lstm_cell_80_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_80/BiasAdd/ReadVariableOp)while/lstm_cell_80/BiasAdd/ReadVariableOp2T
(while/lstm_cell_80/MatMul/ReadVariableOp(while/lstm_cell_80/MatMul/ReadVariableOp2X
*while/lstm_cell_80/MatMul_1/ReadVariableOp*while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�K
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293747

inputs>
+lstm_cell_80_matmul_readvariableop_resource:	�A
-lstm_cell_80_matmul_1_readvariableop_resource:
��;
,lstm_cell_80_biasadd_readvariableop_resource:	�
identity��#lstm_cell_80/BiasAdd/ReadVariableOp�"lstm_cell_80/MatMul/ReadVariableOp�$lstm_cell_80/MatMul_1/ReadVariableOp�while;
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_80/MatMul/ReadVariableOpReadVariableOp+lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_80/MatMulMatMulstrided_slice_2:output:0*lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_80/MatMul_1MatMulzeros:output:0,lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/addAddV2lstm_cell_80/MatMul:product:0lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_80/BiasAddBiasAddlstm_cell_80/add:z:0+lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_80/splitSplit%lstm_cell_80/split/split_dim:output:0lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_80/SigmoidSigmoidlstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_1Sigmoidlstm_cell_80/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_80/mulMullstm_cell_80/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_80/ReluRelulstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_1Mullstm_cell_80/Sigmoid:y:0lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_80/add_1AddV2lstm_cell_80/mul:z:0lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_2Sigmoidlstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_80/Relu_1Relulstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_2Mullstm_cell_80/Sigmoid_2:y:0!lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_80_matmul_readvariableop_resource-lstm_cell_80_matmul_1_readvariableop_resource,lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22293662*
condR
while_cond_22293661*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp$^lstm_cell_80/BiasAdd/ReadVariableOp#^lstm_cell_80/MatMul/ReadVariableOp%^lstm_cell_80/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_80/BiasAdd/ReadVariableOp#lstm_cell_80/BiasAdd/ReadVariableOp2H
"lstm_cell_80/MatMul/ReadVariableOp"lstm_cell_80/MatMul/ReadVariableOp2L
$lstm_cell_80/MatMul_1/ReadVariableOp$lstm_cell_80/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_lstm_79_layer_call_fn_22293145
inputs_0
unknown:	�
	unknown_0:
��
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292262p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�9
�
while_body_22292337
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_80_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_80_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_80_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_80_matmul_readvariableop_resource:	�G
3while_lstm_cell_80_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_80_biasadd_readvariableop_resource:	���)while/lstm_cell_80/BiasAdd/ReadVariableOp�(while/lstm_cell_80/MatMul/ReadVariableOp�*while/lstm_cell_80/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_80/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_80_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_80/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_80_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_80/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/addAddV2#while/lstm_cell_80/MatMul:product:0%while/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_80_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_80/BiasAddBiasAddwhile/lstm_cell_80/add:z:01while/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_80/splitSplit+while/lstm_cell_80/split/split_dim:output:0#while/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_80/SigmoidSigmoid!while/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_1Sigmoid!while/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mulMul while/lstm_cell_80/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_80/ReluRelu!while/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_1Mulwhile/lstm_cell_80/Sigmoid:y:0%while/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/add_1AddV2while/lstm_cell_80/mul:z:0while/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_80/Sigmoid_2Sigmoid!while/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_80/Relu_1Reluwhile/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_80/mul_2Mul while/lstm_cell_80/Sigmoid_2:y:0'while/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_80/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
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
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_80/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_80/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_80/BiasAdd/ReadVariableOp)^while/lstm_cell_80/MatMul/ReadVariableOp+^while/lstm_cell_80/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_80_biasadd_readvariableop_resource4while_lstm_cell_80_biasadd_readvariableop_resource_0"l
3while_lstm_cell_80_matmul_1_readvariableop_resource5while_lstm_cell_80_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_80_matmul_readvariableop_resource3while_lstm_cell_80_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_80/BiasAdd/ReadVariableOp)while/lstm_cell_80/BiasAdd/ReadVariableOp2T
(while/lstm_cell_80/MatMul/ReadVariableOp(while/lstm_cell_80/MatMul/ReadVariableOp2X
*while/lstm_cell_80/MatMul_1/ReadVariableOp*while/lstm_cell_80/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�K
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293602

inputs>
+lstm_cell_80_matmul_readvariableop_resource:	�A
-lstm_cell_80_matmul_1_readvariableop_resource:
��;
,lstm_cell_80_biasadd_readvariableop_resource:	�
identity��#lstm_cell_80/BiasAdd/ReadVariableOp�"lstm_cell_80/MatMul/ReadVariableOp�$lstm_cell_80/MatMul_1/ReadVariableOp�while;
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_80/MatMul/ReadVariableOpReadVariableOp+lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_80/MatMulMatMulstrided_slice_2:output:0*lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_80/MatMul_1MatMulzeros:output:0,lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/addAddV2lstm_cell_80/MatMul:product:0lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_80/BiasAddBiasAddlstm_cell_80/add:z:0+lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_80/splitSplit%lstm_cell_80/split/split_dim:output:0lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_80/SigmoidSigmoidlstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_1Sigmoidlstm_cell_80/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_80/mulMullstm_cell_80/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_80/ReluRelulstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_1Mullstm_cell_80/Sigmoid:y:0lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_80/add_1AddV2lstm_cell_80/mul:z:0lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_2Sigmoidlstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_80/Relu_1Relulstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_2Mullstm_cell_80/Sigmoid_2:y:0!lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_80_matmul_readvariableop_resource-lstm_cell_80_matmul_1_readvariableop_resource,lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22293517*
condR
while_cond_22293516*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp$^lstm_cell_80/BiasAdd/ReadVariableOp#^lstm_cell_80/MatMul/ReadVariableOp%^lstm_cell_80/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_80/BiasAdd/ReadVariableOp#lstm_cell_80/BiasAdd/ReadVariableOp2H
"lstm_cell_80/MatMul/ReadVariableOp"lstm_cell_80/MatMul/ReadVariableOp2L
$lstm_cell_80/MatMul_1/ReadVariableOp$lstm_cell_80/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�3
�	
!__inference__traced_save_22293977
file_prefix.
*savev2_dense_61_kernel_read_readvariableop,
(savev2_dense_61_bias_read_readvariableop:
6savev2_lstm_79_lstm_cell_80_kernel_read_readvariableopD
@savev2_lstm_79_lstm_cell_80_recurrent_kernel_read_readvariableop8
4savev2_lstm_79_lstm_cell_80_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_lstm_79_lstm_cell_80_kernel_read_readvariableopA
=savev2_adam_v_lstm_79_lstm_cell_80_kernel_read_readvariableopK
Gsavev2_adam_m_lstm_79_lstm_cell_80_recurrent_kernel_read_readvariableopK
Gsavev2_adam_v_lstm_79_lstm_cell_80_recurrent_kernel_read_readvariableop?
;savev2_adam_m_lstm_79_lstm_cell_80_bias_read_readvariableop?
;savev2_adam_v_lstm_79_lstm_cell_80_bias_read_readvariableop5
1savev2_adam_m_dense_61_kernel_read_readvariableop5
1savev2_adam_v_dense_61_kernel_read_readvariableop3
/savev2_adam_m_dense_61_bias_read_readvariableop3
/savev2_adam_v_dense_61_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_61_kernel_read_readvariableop(savev2_dense_61_bias_read_readvariableop6savev2_lstm_79_lstm_cell_80_kernel_read_readvariableop@savev2_lstm_79_lstm_cell_80_recurrent_kernel_read_readvariableop4savev2_lstm_79_lstm_cell_80_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_lstm_79_lstm_cell_80_kernel_read_readvariableop=savev2_adam_v_lstm_79_lstm_cell_80_kernel_read_readvariableopGsavev2_adam_m_lstm_79_lstm_cell_80_recurrent_kernel_read_readvariableopGsavev2_adam_v_lstm_79_lstm_cell_80_recurrent_kernel_read_readvariableop;savev2_adam_m_lstm_79_lstm_cell_80_bias_read_readvariableop;savev2_adam_v_lstm_79_lstm_cell_80_bias_read_readvariableop1savev2_adam_m_dense_61_kernel_read_readvariableop1savev2_adam_v_dense_61_kernel_read_readvariableop/savev2_adam_m_dense_61_bias_read_readvariableop/savev2_adam_v_dense_61_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *$
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�::	�:
��:�: : :	�:	�:
��:
��:�:�:	�:	�::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%	!

_output_shapes
:	�:&
"
 
_output_shapes
:
��:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	�: 
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
�
�
while_cond_22292572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22292572___redundant_placeholder06
2while_while_cond_22292572___redundant_placeholder16
2while_while_cond_22292572___redundant_placeholder26
2while_while_cond_22292572___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :����������:����������: ::::: 
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
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�\
�
$__inference__traced_restore_22294050
file_prefix3
 assignvariableop_dense_61_kernel:	�.
 assignvariableop_1_dense_61_bias:A
.assignvariableop_2_lstm_79_lstm_cell_80_kernel:	�L
8assignvariableop_3_lstm_79_lstm_cell_80_recurrent_kernel:
��;
,assignvariableop_4_lstm_79_lstm_cell_80_bias:	�&
assignvariableop_5_iteration:	 *
 assignvariableop_6_learning_rate: H
5assignvariableop_7_adam_m_lstm_79_lstm_cell_80_kernel:	�H
5assignvariableop_8_adam_v_lstm_79_lstm_cell_80_kernel:	�S
?assignvariableop_9_adam_m_lstm_79_lstm_cell_80_recurrent_kernel:
��T
@assignvariableop_10_adam_v_lstm_79_lstm_cell_80_recurrent_kernel:
��C
4assignvariableop_11_adam_m_lstm_79_lstm_cell_80_bias:	�C
4assignvariableop_12_adam_v_lstm_79_lstm_cell_80_bias:	�=
*assignvariableop_13_adam_m_dense_61_kernel:	�=
*assignvariableop_14_adam_v_dense_61_kernel:	�6
(assignvariableop_15_adam_m_dense_61_bias:6
(assignvariableop_16_adam_v_dense_61_bias:%
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: 
identity_22��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_61_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_61_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_79_lstm_cell_80_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_79_lstm_cell_80_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_79_lstm_cell_80_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_iterationIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_learning_rateIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp5assignvariableop_7_adam_m_lstm_79_lstm_cell_80_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp5assignvariableop_8_adam_v_lstm_79_lstm_cell_80_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp?assignvariableop_9_adam_m_lstm_79_lstm_cell_80_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp@assignvariableop_10_adam_v_lstm_79_lstm_cell_80_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_m_lstm_79_lstm_cell_80_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_v_lstm_79_lstm_cell_80_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_m_dense_61_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_v_dense_61_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_m_dense_61_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_v_dense_61_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: �
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
�K
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293312
inputs_0>
+lstm_cell_80_matmul_readvariableop_resource:	�A
-lstm_cell_80_matmul_1_readvariableop_resource:
��;
,lstm_cell_80_biasadd_readvariableop_resource:	�
identity��#lstm_cell_80/BiasAdd/ReadVariableOp�"lstm_cell_80/MatMul/ReadVariableOp�$lstm_cell_80/MatMul_1/ReadVariableOp�while=
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_80/MatMul/ReadVariableOpReadVariableOp+lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_80/MatMulMatMulstrided_slice_2:output:0*lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_80/MatMul_1MatMulzeros:output:0,lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/addAddV2lstm_cell_80/MatMul:product:0lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_80/BiasAddBiasAddlstm_cell_80/add:z:0+lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_80/splitSplit%lstm_cell_80/split/split_dim:output:0lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_80/SigmoidSigmoidlstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_1Sigmoidlstm_cell_80/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_80/mulMullstm_cell_80/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_80/ReluRelulstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_1Mullstm_cell_80/Sigmoid:y:0lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_80/add_1AddV2lstm_cell_80/mul:z:0lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_2Sigmoidlstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_80/Relu_1Relulstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_2Mullstm_cell_80/Sigmoid_2:y:0!lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_80_matmul_readvariableop_resource-lstm_cell_80_matmul_1_readvariableop_resource,lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22293227*
condR
while_cond_22293226*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp$^lstm_cell_80/BiasAdd/ReadVariableOp#^lstm_cell_80/MatMul/ReadVariableOp%^lstm_cell_80/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_80/BiasAdd/ReadVariableOp#lstm_cell_80/BiasAdd/ReadVariableOp2H
"lstm_cell_80/MatMul/ReadVariableOp"lstm_cell_80/MatMul/ReadVariableOp2L
$lstm_cell_80/MatMul_1/ReadVariableOp$lstm_cell_80/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�K
�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22292658

inputs>
+lstm_cell_80_matmul_readvariableop_resource:	�A
-lstm_cell_80_matmul_1_readvariableop_resource:
��;
,lstm_cell_80_biasadd_readvariableop_resource:	�
identity��#lstm_cell_80/BiasAdd/ReadVariableOp�"lstm_cell_80/MatMul/ReadVariableOp�$lstm_cell_80/MatMul_1/ReadVariableOp�while;
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
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB:�
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
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
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
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
"lstm_cell_80/MatMul/ReadVariableOpReadVariableOp+lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_80/MatMulMatMulstrided_slice_2:output:0*lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_80/MatMul_1MatMulzeros:output:0,lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/addAddV2lstm_cell_80/MatMul:product:0lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_80/BiasAddBiasAddlstm_cell_80/add:z:0+lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_80/splitSplit%lstm_cell_80/split/split_dim:output:0lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_80/SigmoidSigmoidlstm_cell_80/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_1Sigmoidlstm_cell_80/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_80/mulMullstm_cell_80/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_80/ReluRelulstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_1Mullstm_cell_80/Sigmoid:y:0lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_80/add_1AddV2lstm_cell_80/mul:z:0lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_80/Sigmoid_2Sigmoidlstm_cell_80/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_80/Relu_1Relulstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_80/mul_2Mullstm_cell_80/Sigmoid_2:y:0!lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:����������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
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
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_80_matmul_readvariableop_resource-lstm_cell_80_matmul_1_readvariableop_resource,lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_22292573*
condR
while_cond_22292572*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp$^lstm_cell_80/BiasAdd/ReadVariableOp#^lstm_cell_80/MatMul/ReadVariableOp%^lstm_cell_80/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_80/BiasAdd/ReadVariableOp#lstm_cell_80/BiasAdd/ReadVariableOp2H
"lstm_cell_80/MatMul/ReadVariableOp"lstm_cell_80/MatMul/ReadVariableOp2L
$lstm_cell_80/MatMul_1/ReadVariableOp$lstm_cell_80/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
F__inference_dense_61_layer_call_and_return_conditional_losses_22293793

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�q
�
#__inference__wrapped_model_22291917
lstm_79_inputT
Asequential_63_lstm_79_lstm_cell_80_matmul_readvariableop_resource:	�W
Csequential_63_lstm_79_lstm_cell_80_matmul_1_readvariableop_resource:
��Q
Bsequential_63_lstm_79_lstm_cell_80_biasadd_readvariableop_resource:	�H
5sequential_63_dense_61_matmul_readvariableop_resource:	�D
6sequential_63_dense_61_biasadd_readvariableop_resource:
identity��-sequential_63/dense_61/BiasAdd/ReadVariableOp�,sequential_63/dense_61/MatMul/ReadVariableOp�9sequential_63/lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp�8sequential_63/lstm_79/lstm_cell_80/MatMul/ReadVariableOp�:sequential_63/lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp�sequential_63/lstm_79/whileX
sequential_63/lstm_79/ShapeShapelstm_79_input*
T0*
_output_shapes
:s
)sequential_63/lstm_79/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_63/lstm_79/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_63/lstm_79/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_63/lstm_79/strided_sliceStridedSlice$sequential_63/lstm_79/Shape:output:02sequential_63/lstm_79/strided_slice/stack:output:04sequential_63/lstm_79/strided_slice/stack_1:output:04sequential_63/lstm_79/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_63/lstm_79/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
"sequential_63/lstm_79/zeros/packedPack,sequential_63/lstm_79/strided_slice:output:0-sequential_63/lstm_79/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_63/lstm_79/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_63/lstm_79/zerosFill+sequential_63/lstm_79/zeros/packed:output:0*sequential_63/lstm_79/zeros/Const:output:0*
T0*(
_output_shapes
:����������i
&sequential_63/lstm_79/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
$sequential_63/lstm_79/zeros_1/packedPack,sequential_63/lstm_79/strided_slice:output:0/sequential_63/lstm_79/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_63/lstm_79/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_63/lstm_79/zeros_1Fill-sequential_63/lstm_79/zeros_1/packed:output:0,sequential_63/lstm_79/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������y
$sequential_63/lstm_79/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_63/lstm_79/transpose	Transposelstm_79_input-sequential_63/lstm_79/transpose/perm:output:0*
T0*+
_output_shapes
:���������p
sequential_63/lstm_79/Shape_1Shape#sequential_63/lstm_79/transpose:y:0*
T0*
_output_shapes
:u
+sequential_63/lstm_79/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_63/lstm_79/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_63/lstm_79/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_63/lstm_79/strided_slice_1StridedSlice&sequential_63/lstm_79/Shape_1:output:04sequential_63/lstm_79/strided_slice_1/stack:output:06sequential_63/lstm_79/strided_slice_1/stack_1:output:06sequential_63/lstm_79/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_63/lstm_79/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential_63/lstm_79/TensorArrayV2TensorListReserve:sequential_63/lstm_79/TensorArrayV2/element_shape:output:0.sequential_63/lstm_79/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ksequential_63/lstm_79/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
=sequential_63/lstm_79/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_63/lstm_79/transpose:y:0Tsequential_63/lstm_79/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���u
+sequential_63/lstm_79/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_63/lstm_79/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_63/lstm_79/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_63/lstm_79/strided_slice_2StridedSlice#sequential_63/lstm_79/transpose:y:04sequential_63/lstm_79/strided_slice_2/stack:output:06sequential_63/lstm_79/strided_slice_2/stack_1:output:06sequential_63/lstm_79/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
8sequential_63/lstm_79/lstm_cell_80/MatMul/ReadVariableOpReadVariableOpAsequential_63_lstm_79_lstm_cell_80_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
)sequential_63/lstm_79/lstm_cell_80/MatMulMatMul.sequential_63/lstm_79/strided_slice_2:output:0@sequential_63/lstm_79/lstm_cell_80/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:sequential_63/lstm_79/lstm_cell_80/MatMul_1/ReadVariableOpReadVariableOpCsequential_63_lstm_79_lstm_cell_80_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
+sequential_63/lstm_79/lstm_cell_80/MatMul_1MatMul$sequential_63/lstm_79/zeros:output:0Bsequential_63/lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&sequential_63/lstm_79/lstm_cell_80/addAddV23sequential_63/lstm_79/lstm_cell_80/MatMul:product:05sequential_63/lstm_79/lstm_cell_80/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
9sequential_63/lstm_79/lstm_cell_80/BiasAdd/ReadVariableOpReadVariableOpBsequential_63_lstm_79_lstm_cell_80_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*sequential_63/lstm_79/lstm_cell_80/BiasAddBiasAdd*sequential_63/lstm_79/lstm_cell_80/add:z:0Asequential_63/lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
2sequential_63/lstm_79/lstm_cell_80/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(sequential_63/lstm_79/lstm_cell_80/splitSplit;sequential_63/lstm_79/lstm_cell_80/split/split_dim:output:03sequential_63/lstm_79/lstm_cell_80/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
*sequential_63/lstm_79/lstm_cell_80/SigmoidSigmoid1sequential_63/lstm_79/lstm_cell_80/split:output:0*
T0*(
_output_shapes
:�����������
,sequential_63/lstm_79/lstm_cell_80/Sigmoid_1Sigmoid1sequential_63/lstm_79/lstm_cell_80/split:output:1*
T0*(
_output_shapes
:�����������
&sequential_63/lstm_79/lstm_cell_80/mulMul0sequential_63/lstm_79/lstm_cell_80/Sigmoid_1:y:0&sequential_63/lstm_79/zeros_1:output:0*
T0*(
_output_shapes
:�����������
'sequential_63/lstm_79/lstm_cell_80/ReluRelu1sequential_63/lstm_79/lstm_cell_80/split:output:2*
T0*(
_output_shapes
:�����������
(sequential_63/lstm_79/lstm_cell_80/mul_1Mul.sequential_63/lstm_79/lstm_cell_80/Sigmoid:y:05sequential_63/lstm_79/lstm_cell_80/Relu:activations:0*
T0*(
_output_shapes
:�����������
(sequential_63/lstm_79/lstm_cell_80/add_1AddV2*sequential_63/lstm_79/lstm_cell_80/mul:z:0,sequential_63/lstm_79/lstm_cell_80/mul_1:z:0*
T0*(
_output_shapes
:�����������
,sequential_63/lstm_79/lstm_cell_80/Sigmoid_2Sigmoid1sequential_63/lstm_79/lstm_cell_80/split:output:3*
T0*(
_output_shapes
:�����������
)sequential_63/lstm_79/lstm_cell_80/Relu_1Relu,sequential_63/lstm_79/lstm_cell_80/add_1:z:0*
T0*(
_output_shapes
:�����������
(sequential_63/lstm_79/lstm_cell_80/mul_2Mul0sequential_63/lstm_79/lstm_cell_80/Sigmoid_2:y:07sequential_63/lstm_79/lstm_cell_80/Relu_1:activations:0*
T0*(
_output_shapes
:�����������
3sequential_63/lstm_79/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   t
2sequential_63/lstm_79/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
%sequential_63/lstm_79/TensorArrayV2_1TensorListReserve<sequential_63/lstm_79/TensorArrayV2_1/element_shape:output:0;sequential_63/lstm_79/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���\
sequential_63/lstm_79/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_63/lstm_79/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
(sequential_63/lstm_79/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_63/lstm_79/whileWhile1sequential_63/lstm_79/while/loop_counter:output:07sequential_63/lstm_79/while/maximum_iterations:output:0#sequential_63/lstm_79/time:output:0.sequential_63/lstm_79/TensorArrayV2_1:handle:0$sequential_63/lstm_79/zeros:output:0&sequential_63/lstm_79/zeros_1:output:0.sequential_63/lstm_79/strided_slice_1:output:0Msequential_63/lstm_79/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_63_lstm_79_lstm_cell_80_matmul_readvariableop_resourceCsequential_63_lstm_79_lstm_cell_80_matmul_1_readvariableop_resourceBsequential_63_lstm_79_lstm_cell_80_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_63_lstm_79_while_body_22291825*5
cond-R+
)sequential_63_lstm_79_while_cond_22291824*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
Fsequential_63/lstm_79/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
8sequential_63/lstm_79/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_63/lstm_79/while:output:3Osequential_63/lstm_79/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elements~
+sequential_63/lstm_79/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������w
-sequential_63/lstm_79/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_63/lstm_79/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_63/lstm_79/strided_slice_3StridedSliceAsequential_63/lstm_79/TensorArrayV2Stack/TensorListStack:tensor:04sequential_63/lstm_79/strided_slice_3/stack:output:06sequential_63/lstm_79/strided_slice_3/stack_1:output:06sequential_63/lstm_79/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask{
&sequential_63/lstm_79/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
!sequential_63/lstm_79/transpose_1	TransposeAsequential_63/lstm_79/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_63/lstm_79/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������q
sequential_63/lstm_79/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
!sequential_63/dropout_44/IdentityIdentity.sequential_63/lstm_79/strided_slice_3:output:0*
T0*(
_output_shapes
:�����������
,sequential_63/dense_61/MatMul/ReadVariableOpReadVariableOp5sequential_63_dense_61_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_63/dense_61/MatMulMatMul*sequential_63/dropout_44/Identity:output:04sequential_63/dense_61/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_63/dense_61/BiasAdd/ReadVariableOpReadVariableOp6sequential_63_dense_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_63/dense_61/BiasAddBiasAdd'sequential_63/dense_61/MatMul:product:05sequential_63/dense_61/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_63/dense_61/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_63/dense_61/BiasAdd/ReadVariableOp-^sequential_63/dense_61/MatMul/ReadVariableOp:^sequential_63/lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp9^sequential_63/lstm_79/lstm_cell_80/MatMul/ReadVariableOp;^sequential_63/lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp^sequential_63/lstm_79/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2^
-sequential_63/dense_61/BiasAdd/ReadVariableOp-sequential_63/dense_61/BiasAdd/ReadVariableOp2\
,sequential_63/dense_61/MatMul/ReadVariableOp,sequential_63/dense_61/MatMul/ReadVariableOp2v
9sequential_63/lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp9sequential_63/lstm_79/lstm_cell_80/BiasAdd/ReadVariableOp2t
8sequential_63/lstm_79/lstm_cell_80/MatMul/ReadVariableOp8sequential_63/lstm_79/lstm_cell_80/MatMul/ReadVariableOp2x
:sequential_63/lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp:sequential_63/lstm_79/lstm_cell_80/MatMul_1/ReadVariableOp2:
sequential_63/lstm_79/whilesequential_63/lstm_79/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_79_input"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
lstm_79_input:
serving_default_lstm_79_input:0���������<
dense_610
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
�
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
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
�
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
�
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
�
-trace_0
.trace_1
/trace_2
0trace_32�
0__inference_sequential_63_layer_call_fn_22292467
0__inference_sequential_63_layer_call_fn_22292797
0__inference_sequential_63_layer_call_fn_22292812
0__inference_sequential_63_layer_call_fn_22292729�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z-trace_0z.trace_1z/trace_2z0trace_3
�
1trace_0
2trace_1
3trace_2
4trace_32�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292964
K__inference_sequential_63_layer_call_and_return_conditional_losses_22293123
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292746
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292763�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z1trace_0z2trace_1z3trace_2z4trace_3
�B�
#__inference__wrapped_model_22291917lstm_79_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
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
�

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
�
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32�
*__inference_lstm_79_layer_call_fn_22293134
*__inference_lstm_79_layer_call_fn_22293145
*__inference_lstm_79_layer_call_fn_22293156
*__inference_lstm_79_layer_call_fn_22293167�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
�
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293312
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293457
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293602
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293747�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
"
_generic_user_object
�
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
�
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
�
Xtrace_0
Ytrace_12�
-__inference_dropout_44_layer_call_fn_22293752
-__inference_dropout_44_layer_call_fn_22293757�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0zYtrace_1
�
Ztrace_0
[trace_12�
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293762
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293774�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
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
�
atrace_02�
+__inference_dense_61_layer_call_fn_22293783�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
�
btrace_02�
F__inference_dense_61_layer_call_and_return_conditional_losses_22293793�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
": 	�2dense_61/kernel
:2dense_61/bias
.:,	�2lstm_79/lstm_cell_80/kernel
9:7
��2%lstm_79/lstm_cell_80/recurrent_kernel
(:&�2lstm_79/lstm_cell_80/bias
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
�B�
0__inference_sequential_63_layer_call_fn_22292467lstm_79_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_sequential_63_layer_call_fn_22292797inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_sequential_63_layer_call_fn_22292812inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_sequential_63_layer_call_fn_22292729lstm_79_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292964inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22293123inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292746lstm_79_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292763lstm_79_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�2��
���
FullArgSpec2
args*�'
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_22292782lstm_79_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_lstm_79_layer_call_fn_22293134inputs_0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_lstm_79_layer_call_fn_22293145inputs_0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_lstm_79_layer_call_fn_22293156inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_lstm_79_layer_call_fn_22293167inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293312inputs_0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293457inputs_0"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293602inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293747inputs"�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
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
�
ttrace_0
utrace_12�
/__inference_lstm_cell_80_layer_call_fn_22293810
/__inference_lstm_cell_80_layer_call_fn_22293827�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zttrace_0zutrace_1
�
vtrace_0
wtrace_12�
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293859
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293891�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_dropout_44_layer_call_fn_22293752inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_44_layer_call_fn_22293757inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293762inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293774inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_dense_61_layer_call_fn_22293783inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_61_layer_call_and_return_conditional_losses_22293793inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�
_fn_kwargs"
_tf_keras_metric
3:1	�2"Adam/m/lstm_79/lstm_cell_80/kernel
3:1	�2"Adam/v/lstm_79/lstm_cell_80/kernel
>:<
��2,Adam/m/lstm_79/lstm_cell_80/recurrent_kernel
>:<
��2,Adam/v/lstm_79/lstm_cell_80/recurrent_kernel
-:+�2 Adam/m/lstm_79/lstm_cell_80/bias
-:+�2 Adam/v/lstm_79/lstm_cell_80/bias
':%	�2Adam/m/dense_61/kernel
':%	�2Adam/v/dense_61/kernel
 :2Adam/m/dense_61/bias
 :2Adam/v/dense_61/bias
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
�B�
/__inference_lstm_cell_80_layer_call_fn_22293810inputsstates_0states_1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_lstm_cell_80_layer_call_fn_22293827inputsstates_0states_1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293859inputsstates_0states_1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293891inputsstates_0states_1"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
trackable_dict_wrapper�
#__inference__wrapped_model_22291917x%&'#$:�7
0�-
+�(
lstm_79_input���������
� "3�0
.
dense_61"�
dense_61����������
F__inference_dense_61_layer_call_and_return_conditional_losses_22293793d#$0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_61_layer_call_fn_22293783Y#$0�-
&�#
!�
inputs����������
� "!�
unknown����������
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293762e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
H__inference_dropout_44_layer_call_and_return_conditional_losses_22293774e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
-__inference_dropout_44_layer_call_fn_22293752Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
-__inference_dropout_44_layer_call_fn_22293757Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293312�%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "-�*
#� 
tensor_0����������
� �
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293457�%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "-�*
#� 
tensor_0����������
� �
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293602u%&'?�<
5�2
$�!
inputs���������

 
p 

 
� "-�*
#� 
tensor_0����������
� �
E__inference_lstm_79_layer_call_and_return_conditional_losses_22293747u%&'?�<
5�2
$�!
inputs���������

 
p

 
� "-�*
#� 
tensor_0����������
� �
*__inference_lstm_79_layer_call_fn_22293134z%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ""�
unknown�����������
*__inference_lstm_79_layer_call_fn_22293145z%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ""�
unknown�����������
*__inference_lstm_79_layer_call_fn_22293156j%&'?�<
5�2
$�!
inputs���������

 
p 

 
� ""�
unknown�����������
*__inference_lstm_79_layer_call_fn_22293167j%&'?�<
5�2
$�!
inputs���������

 
p

 
� ""�
unknown�����������
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293859�%&'��
x�u
 �
inputs���������
M�J
#� 
states_0����������
#� 
states_1����������
p 
� "���
��~
%�"

tensor_0_0����������
U�R
'�$
tensor_0_1_0����������
'�$
tensor_0_1_1����������
� �
J__inference_lstm_cell_80_layer_call_and_return_conditional_losses_22293891�%&'��
x�u
 �
inputs���������
M�J
#� 
states_0����������
#� 
states_1����������
p
� "���
��~
%�"

tensor_0_0����������
U�R
'�$
tensor_0_1_0����������
'�$
tensor_0_1_1����������
� �
/__inference_lstm_cell_80_layer_call_fn_22293810�%&'��
x�u
 �
inputs���������
M�J
#� 
states_0����������
#� 
states_1����������
p 
� "{�x
#� 
tensor_0����������
Q�N
%�"

tensor_1_0����������
%�"

tensor_1_1�����������
/__inference_lstm_cell_80_layer_call_fn_22293827�%&'��
x�u
 �
inputs���������
M�J
#� 
states_0����������
#� 
states_1����������
p
� "{�x
#� 
tensor_0����������
Q�N
%�"

tensor_1_0����������
%�"

tensor_1_1�����������
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292746y%&'#$B�?
8�5
+�(
lstm_79_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292763y%&'#$B�?
8�5
+�(
lstm_79_input���������
p

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_63_layer_call_and_return_conditional_losses_22292964r%&'#$;�8
1�.
$�!
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_63_layer_call_and_return_conditional_losses_22293123r%&'#$;�8
1�.
$�!
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
0__inference_sequential_63_layer_call_fn_22292467n%&'#$B�?
8�5
+�(
lstm_79_input���������
p 

 
� "!�
unknown����������
0__inference_sequential_63_layer_call_fn_22292729n%&'#$B�?
8�5
+�(
lstm_79_input���������
p

 
� "!�
unknown����������
0__inference_sequential_63_layer_call_fn_22292797g%&'#$;�8
1�.
$�!
inputs���������
p 

 
� "!�
unknown����������
0__inference_sequential_63_layer_call_fn_22292812g%&'#$;�8
1�.
$�!
inputs���������
p

 
� "!�
unknown����������
&__inference_signature_wrapper_22292782�%&'#$K�H
� 
A�>
<
lstm_79_input+�(
lstm_79_input���������"3�0
.
dense_61"�
dense_61���������