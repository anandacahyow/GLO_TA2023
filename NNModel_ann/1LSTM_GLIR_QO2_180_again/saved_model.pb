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
Adam/v/dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_74/bias
y
(Adam/v/dense_74/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_74/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_74/bias
y
(Adam/m/dense_74/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_74/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/v/dense_74/kernel
�
*Adam/v/dense_74/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_74/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*'
shared_nameAdam/m/dense_74/kernel
�
*Adam/m/dense_74/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_74/kernel*
_output_shapes
:	�*
dtype0
�
 Adam/v/lstm_92/lstm_cell_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/v/lstm_92/lstm_cell_96/bias
�
4Adam/v/lstm_92/lstm_cell_96/bias/Read/ReadVariableOpReadVariableOp Adam/v/lstm_92/lstm_cell_96/bias*
_output_shapes	
:�*
dtype0
�
 Adam/m/lstm_92/lstm_cell_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/m/lstm_92/lstm_cell_96/bias
�
4Adam/m/lstm_92/lstm_cell_96/bias/Read/ReadVariableOpReadVariableOp Adam/m/lstm_92/lstm_cell_96/bias*
_output_shapes	
:�*
dtype0
�
,Adam/v/lstm_92/lstm_cell_96/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*=
shared_name.,Adam/v/lstm_92/lstm_cell_96/recurrent_kernel
�
@Adam/v/lstm_92/lstm_cell_96/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/v/lstm_92/lstm_cell_96/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
,Adam/m/lstm_92/lstm_cell_96/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*=
shared_name.,Adam/m/lstm_92/lstm_cell_96/recurrent_kernel
�
@Adam/m/lstm_92/lstm_cell_96/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/m/lstm_92/lstm_cell_96/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
"Adam/v/lstm_92/lstm_cell_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/v/lstm_92/lstm_cell_96/kernel
�
6Adam/v/lstm_92/lstm_cell_96/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_92/lstm_cell_96/kernel*
_output_shapes
:	�*
dtype0
�
"Adam/m/lstm_92/lstm_cell_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/m/lstm_92/lstm_cell_96/kernel
�
6Adam/m/lstm_92/lstm_cell_96/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_92/lstm_cell_96/kernel*
_output_shapes
:	�*
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
lstm_92/lstm_cell_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_92/lstm_cell_96/bias
�
-lstm_92/lstm_cell_96/bias/Read/ReadVariableOpReadVariableOplstm_92/lstm_cell_96/bias*
_output_shapes	
:�*
dtype0
�
%lstm_92/lstm_cell_96/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*6
shared_name'%lstm_92/lstm_cell_96/recurrent_kernel
�
9lstm_92/lstm_cell_96/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_92/lstm_cell_96/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
lstm_92/lstm_cell_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_92/lstm_cell_96/kernel
�
/lstm_92/lstm_cell_96/kernel/Read/ReadVariableOpReadVariableOplstm_92/lstm_cell_96/kernel*
_output_shapes
:	�*
dtype0
r
dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_74/bias
k
!dense_74/bias/Read/ReadVariableOpReadVariableOpdense_74/bias*
_output_shapes
:*
dtype0
{
dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_74/kernel
t
#dense_74/kernel/Read/ReadVariableOpReadVariableOpdense_74/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_lstm_92_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_92_inputlstm_92/lstm_cell_96/kernel%lstm_92/lstm_cell_96/recurrent_kernellstm_92/lstm_cell_96/biasdense_74/kerneldense_74/bias*
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
&__inference_signature_wrapper_22696369

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
VARIABLE_VALUEdense_74/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_74/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_92/lstm_cell_96/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_92/lstm_cell_96/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_92/lstm_cell_96/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE"Adam/m/lstm_92/lstm_cell_96/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_92/lstm_cell_96/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/m/lstm_92/lstm_cell_96/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/v/lstm_92/lstm_cell_96/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_92/lstm_cell_96/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_92/lstm_cell_96/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_74/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_74/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_74/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_74/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_74/kernel/Read/ReadVariableOp!dense_74/bias/Read/ReadVariableOp/lstm_92/lstm_cell_96/kernel/Read/ReadVariableOp9lstm_92/lstm_cell_96/recurrent_kernel/Read/ReadVariableOp-lstm_92/lstm_cell_96/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/lstm_92/lstm_cell_96/kernel/Read/ReadVariableOp6Adam/v/lstm_92/lstm_cell_96/kernel/Read/ReadVariableOp@Adam/m/lstm_92/lstm_cell_96/recurrent_kernel/Read/ReadVariableOp@Adam/v/lstm_92/lstm_cell_96/recurrent_kernel/Read/ReadVariableOp4Adam/m/lstm_92/lstm_cell_96/bias/Read/ReadVariableOp4Adam/v/lstm_92/lstm_cell_96/bias/Read/ReadVariableOp*Adam/m/dense_74/kernel/Read/ReadVariableOp*Adam/v/dense_74/kernel/Read/ReadVariableOp(Adam/m/dense_74/bias/Read/ReadVariableOp(Adam/v/dense_74/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*"
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
!__inference__traced_save_22697564
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_74/kerneldense_74/biaslstm_92/lstm_cell_96/kernel%lstm_92/lstm_cell_96/recurrent_kernellstm_92/lstm_cell_96/bias	iterationlearning_rate"Adam/m/lstm_92/lstm_cell_96/kernel"Adam/v/lstm_92/lstm_cell_96/kernel,Adam/m/lstm_92/lstm_cell_96/recurrent_kernel,Adam/v/lstm_92/lstm_cell_96/recurrent_kernel Adam/m/lstm_92/lstm_cell_96/bias Adam/v/lstm_92/lstm_cell_96/biasAdam/m/dense_74/kernelAdam/v/dense_74/kernelAdam/m/dense_74/biasAdam/v/dense_74/biastotal_1count_1totalcount*!
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
$__inference__traced_restore_22697637��
�
�
/__inference_lstm_cell_96_layer_call_fn_22697397

inputs
states_0
states_1
unknown:	�
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695571p
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
A:���������:����������:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
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

�
lstm_92_while_cond_22696458,
(lstm_92_while_lstm_92_while_loop_counter2
.lstm_92_while_lstm_92_while_maximum_iterations
lstm_92_while_placeholder
lstm_92_while_placeholder_1
lstm_92_while_placeholder_2
lstm_92_while_placeholder_3.
*lstm_92_while_less_lstm_92_strided_slice_1F
Blstm_92_while_lstm_92_while_cond_22696458___redundant_placeholder0F
Blstm_92_while_lstm_92_while_cond_22696458___redundant_placeholder1F
Blstm_92_while_lstm_92_while_cond_22696458___redundant_placeholder2F
Blstm_92_while_lstm_92_while_cond_22696458___redundant_placeholder3
lstm_92_while_identity
�
lstm_92/while/LessLesslstm_92_while_placeholder*lstm_92_while_less_lstm_92_strided_slice_1*
T0*
_output_shapes
: [
lstm_92/while/IdentityIdentitylstm_92/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_92_while_identitylstm_92/while/Identity:output:0*(
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
�
�
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696288

inputs#
lstm_92_22696274:	�$
lstm_92_22696276:
��
lstm_92_22696278:	�$
dense_74_22696282:	�
dense_74_22696284:
identity�� dense_74/StatefulPartitionedCall�"dropout_57/StatefulPartitionedCall�lstm_92/StatefulPartitionedCall�
lstm_92/StatefulPartitionedCallStatefulPartitionedCallinputslstm_92_22696274lstm_92_22696276lstm_92_22696278*
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696245�
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall(lstm_92/StatefulPartitionedCall:output:0*
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696084�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall+dropout_57/StatefulPartitionedCall:output:0dense_74_22696282dense_74_22696284*
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
F__inference_dense_74_layer_call_and_return_conditional_losses_22696034x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_74/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall ^lstm_92/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2B
lstm_92/StatefulPartitionedCalllstm_92/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695571

inputs

states
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�K
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696009

inputs>
+lstm_cell_96_matmul_readvariableop_resource:	�A
-lstm_cell_96_matmul_1_readvariableop_resource:
��;
,lstm_cell_96_biasadd_readvariableop_resource:	�
identity��#lstm_cell_96/BiasAdd/ReadVariableOp�"lstm_cell_96/MatMul/ReadVariableOp�$lstm_cell_96/MatMul_1/ReadVariableOp�while;
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
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
"lstm_cell_96/MatMul/ReadVariableOpReadVariableOp+lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_96/MatMulMatMulstrided_slice_2:output:0*lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_96/MatMul_1MatMulzeros:output:0,lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/addAddV2lstm_cell_96/MatMul:product:0lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_96/BiasAddBiasAddlstm_cell_96/add:z:0+lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_96/splitSplit%lstm_cell_96/split/split_dim:output:0lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_96/SigmoidSigmoidlstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_1Sigmoidlstm_cell_96/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_96/mulMullstm_cell_96/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_96/ReluRelulstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_1Mullstm_cell_96/Sigmoid:y:0lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_96/add_1AddV2lstm_cell_96/mul:z:0lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_2Sigmoidlstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_96/Relu_1Relulstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_2Mullstm_cell_96/Sigmoid_2:y:0!lstm_cell_96/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_96_matmul_readvariableop_resource-lstm_cell_96_matmul_1_readvariableop_resource,lstm_cell_96_biasadd_readvariableop_resource*
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
while_body_22695924*
condR
while_cond_22695923*M
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
NoOpNoOp$^lstm_cell_96/BiasAdd/ReadVariableOp#^lstm_cell_96/MatMul/ReadVariableOp%^lstm_cell_96/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_96/BiasAdd/ReadVariableOp#lstm_cell_96/BiasAdd/ReadVariableOp2H
"lstm_cell_96/MatMul/ReadVariableOp"lstm_cell_96/MatMul/ReadVariableOp2L
$lstm_cell_96/MatMul_1/ReadVariableOp$lstm_cell_96/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_22695924
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_96_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_96_matmul_readvariableop_resource:	�G
3while_lstm_cell_96_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_96_biasadd_readvariableop_resource:	���)while/lstm_cell_96/BiasAdd/ReadVariableOp�(while/lstm_cell_96/MatMul/ReadVariableOp�*while/lstm_cell_96/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_96/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_96/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/addAddV2#while/lstm_cell_96/MatMul:product:0%while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_96/BiasAddBiasAddwhile/lstm_cell_96/add:z:01while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_96/splitSplit+while/lstm_cell_96/split/split_dim:output:0#while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_96/SigmoidSigmoid!while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_1Sigmoid!while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mulMul while/lstm_cell_96/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_96/ReluRelu!while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_1Mulwhile/lstm_cell_96/Sigmoid:y:0%while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/add_1AddV2while/lstm_cell_96/mul:z:0while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_2Sigmoid!while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_96/Relu_1Reluwhile/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_2Mul while/lstm_cell_96/Sigmoid_2:y:0'while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_96/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_96/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_96/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_96/BiasAdd/ReadVariableOp)^while/lstm_cell_96/MatMul/ReadVariableOp+^while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_96_biasadd_readvariableop_resource4while_lstm_cell_96_biasadd_readvariableop_resource_0"l
3while_lstm_cell_96_matmul_1_readvariableop_resource5while_lstm_cell_96_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_96_matmul_readvariableop_resource3while_lstm_cell_96_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_96/BiasAdd/ReadVariableOp)while/lstm_cell_96/BiasAdd/ReadVariableOp2T
(while/lstm_cell_96/MatMul/ReadVariableOp(while/lstm_cell_96/MatMul/ReadVariableOp2X
*while/lstm_cell_96/MatMul_1/ReadVariableOp*while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
while_cond_22696159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22696159___redundant_placeholder06
2while_while_cond_22696159___redundant_placeholder16
2while_while_cond_22696159___redundant_placeholder26
2while_while_cond_22696159___redundant_placeholder3
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
�
�
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695719

inputs

states
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�
�
)sequential_76_lstm_92_while_cond_22695411H
Dsequential_76_lstm_92_while_sequential_76_lstm_92_while_loop_counterN
Jsequential_76_lstm_92_while_sequential_76_lstm_92_while_maximum_iterations+
'sequential_76_lstm_92_while_placeholder-
)sequential_76_lstm_92_while_placeholder_1-
)sequential_76_lstm_92_while_placeholder_2-
)sequential_76_lstm_92_while_placeholder_3J
Fsequential_76_lstm_92_while_less_sequential_76_lstm_92_strided_slice_1b
^sequential_76_lstm_92_while_sequential_76_lstm_92_while_cond_22695411___redundant_placeholder0b
^sequential_76_lstm_92_while_sequential_76_lstm_92_while_cond_22695411___redundant_placeholder1b
^sequential_76_lstm_92_while_sequential_76_lstm_92_while_cond_22695411___redundant_placeholder2b
^sequential_76_lstm_92_while_sequential_76_lstm_92_while_cond_22695411___redundant_placeholder3(
$sequential_76_lstm_92_while_identity
�
 sequential_76/lstm_92/while/LessLess'sequential_76_lstm_92_while_placeholderFsequential_76_lstm_92_while_less_sequential_76_lstm_92_strided_slice_1*
T0*
_output_shapes
: w
$sequential_76/lstm_92/while/IdentityIdentity$sequential_76/lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_76_lstm_92_while_identity-sequential_76/lstm_92/while/Identity:output:0*(
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
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22695656

inputs(
lstm_cell_96_22695572:	�)
lstm_cell_96_22695574:
��$
lstm_cell_96_22695576:	�
identity��$lstm_cell_96/StatefulPartitionedCall�while;
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
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
$lstm_cell_96/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_96_22695572lstm_cell_96_22695574lstm_cell_96_22695576*
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695571n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_96_22695572lstm_cell_96_22695574lstm_cell_96_22695576*
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
while_body_22695586*
condR
while_cond_22695585*M
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
NoOpNoOp%^lstm_cell_96/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_96/StatefulPartitionedCall$lstm_cell_96/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�9
�
while_body_22696160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_96_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_96_matmul_readvariableop_resource:	�G
3while_lstm_cell_96_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_96_biasadd_readvariableop_resource:	���)while/lstm_cell_96/BiasAdd/ReadVariableOp�(while/lstm_cell_96/MatMul/ReadVariableOp�*while/lstm_cell_96/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_96/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_96/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/addAddV2#while/lstm_cell_96/MatMul:product:0%while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_96/BiasAddBiasAddwhile/lstm_cell_96/add:z:01while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_96/splitSplit+while/lstm_cell_96/split/split_dim:output:0#while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_96/SigmoidSigmoid!while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_1Sigmoid!while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mulMul while/lstm_cell_96/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_96/ReluRelu!while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_1Mulwhile/lstm_cell_96/Sigmoid:y:0%while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/add_1AddV2while/lstm_cell_96/mul:z:0while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_2Sigmoid!while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_96/Relu_1Reluwhile/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_2Mul while/lstm_cell_96/Sigmoid_2:y:0'while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_96/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_96/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_96/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_96/BiasAdd/ReadVariableOp)^while/lstm_cell_96/MatMul/ReadVariableOp+^while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_96_biasadd_readvariableop_resource4while_lstm_cell_96_biasadd_readvariableop_resource_0"l
3while_lstm_cell_96_matmul_1_readvariableop_resource5while_lstm_cell_96_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_96_matmul_readvariableop_resource3while_lstm_cell_96_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_96/BiasAdd/ReadVariableOp)while/lstm_cell_96/BiasAdd/ReadVariableOp2T
(while/lstm_cell_96/MatMul/ReadVariableOp(while/lstm_cell_96/MatMul/ReadVariableOp2X
*while/lstm_cell_96/MatMul_1/ReadVariableOp*while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
�9
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22695849

inputs(
lstm_cell_96_22695765:	�)
lstm_cell_96_22695767:
��$
lstm_cell_96_22695769:	�
identity��$lstm_cell_96/StatefulPartitionedCall�while;
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
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
$lstm_cell_96/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_96_22695765lstm_cell_96_22695767lstm_cell_96_22695769*
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695719n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_96_22695765lstm_cell_96_22695767lstm_cell_96_22695769*
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
while_body_22695779*
condR
while_cond_22695778*M
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
NoOpNoOp%^lstm_cell_96/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_96/StatefulPartitionedCall$lstm_cell_96/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�e
�
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696710

inputsF
3lstm_92_lstm_cell_96_matmul_readvariableop_resource:	�I
5lstm_92_lstm_cell_96_matmul_1_readvariableop_resource:
��C
4lstm_92_lstm_cell_96_biasadd_readvariableop_resource:	�:
'dense_74_matmul_readvariableop_resource:	�6
(dense_74_biasadd_readvariableop_resource:
identity��dense_74/BiasAdd/ReadVariableOp�dense_74/MatMul/ReadVariableOp�+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp�*lstm_92/lstm_cell_96/MatMul/ReadVariableOp�,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp�lstm_92/whileC
lstm_92/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_sliceStridedSlicelstm_92/Shape:output:0$lstm_92/strided_slice/stack:output:0&lstm_92/strided_slice/stack_1:output:0&lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_92/zeros/packedPacklstm_92/strided_slice:output:0lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_92/zerosFilllstm_92/zeros/packed:output:0lstm_92/zeros/Const:output:0*
T0*(
_output_shapes
:����������[
lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_92/zeros_1/packedPacklstm_92/strided_slice:output:0!lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_92/zeros_1Filllstm_92/zeros_1/packed:output:0lstm_92/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������k
lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_92/transpose	Transposeinputslstm_92/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_92/Shape_1Shapelstm_92/transpose:y:0*
T0*
_output_shapes
:g
lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_slice_1StridedSlicelstm_92/Shape_1:output:0&lstm_92/strided_slice_1/stack:output:0(lstm_92/strided_slice_1/stack_1:output:0(lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_92/TensorArrayV2TensorListReserve,lstm_92/TensorArrayV2/element_shape:output:0 lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_92/transpose:y:0Flstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_slice_2StridedSlicelstm_92/transpose:y:0&lstm_92/strided_slice_2/stack:output:0(lstm_92/strided_slice_2/stack_1:output:0(lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_92/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3lstm_92_lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_92/lstm_cell_96/MatMulMatMul lstm_92/strided_slice_2:output:02lstm_92/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5lstm_92_lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_92/lstm_cell_96/MatMul_1MatMullstm_92/zeros:output:04lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/addAddV2%lstm_92/lstm_cell_96/MatMul:product:0'lstm_92/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4lstm_92_lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_92/lstm_cell_96/BiasAddBiasAddlstm_92/lstm_cell_96/add:z:03lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_92/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_92/lstm_cell_96/splitSplit-lstm_92/lstm_cell_96/split/split_dim:output:0%lstm_92/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split
lstm_92/lstm_cell_96/SigmoidSigmoid#lstm_92/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/Sigmoid_1Sigmoid#lstm_92/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/mulMul"lstm_92/lstm_cell_96/Sigmoid_1:y:0lstm_92/zeros_1:output:0*
T0*(
_output_shapes
:����������y
lstm_92/lstm_cell_96/ReluRelu#lstm_92/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/mul_1Mul lstm_92/lstm_cell_96/Sigmoid:y:0'lstm_92/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/add_1AddV2lstm_92/lstm_cell_96/mul:z:0lstm_92/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/Sigmoid_2Sigmoid#lstm_92/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������v
lstm_92/lstm_cell_96/Relu_1Relulstm_92/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/mul_2Mul"lstm_92/lstm_cell_96/Sigmoid_2:y:0)lstm_92/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������v
%lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   f
$lstm_92/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_92/TensorArrayV2_1TensorListReserve.lstm_92/TensorArrayV2_1/element_shape:output:0-lstm_92/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_92/whileWhile#lstm_92/while/loop_counter:output:0)lstm_92/while/maximum_iterations:output:0lstm_92/time:output:0 lstm_92/TensorArrayV2_1:handle:0lstm_92/zeros:output:0lstm_92/zeros_1:output:0 lstm_92/strided_slice_1:output:0?lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_92_lstm_cell_96_matmul_readvariableop_resource5lstm_92_lstm_cell_96_matmul_1_readvariableop_resource4lstm_92_lstm_cell_96_biasadd_readvariableop_resource*
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
lstm_92_while_body_22696611*'
condR
lstm_92_while_cond_22696610*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
8lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
*lstm_92/TensorArrayV2Stack/TensorListStackTensorListStacklstm_92/while:output:3Alstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsp
lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_slice_3StridedSlice3lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_92/strided_slice_3/stack:output:0(lstm_92/strided_slice_3/stack_1:output:0(lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskm
lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_92/transpose_1	Transpose3lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_92/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������c
lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_57/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_57/dropout/MulMul lstm_92/strided_slice_3:output:0!dropout_57/dropout/Const:output:0*
T0*(
_output_shapes
:����������h
dropout_57/dropout/ShapeShape lstm_92/strided_slice_3:output:0*
T0*
_output_shapes
:�
/dropout_57/dropout/random_uniform/RandomUniformRandomUniform!dropout_57/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_57/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_57/dropout/GreaterEqualGreaterEqual8dropout_57/dropout/random_uniform/RandomUniform:output:0*dropout_57/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������_
dropout_57/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_57/dropout/SelectV2SelectV2#dropout_57/dropout/GreaterEqual:z:0dropout_57/dropout/Mul:z:0#dropout_57/dropout/Const_1:output:0*
T0*(
_output_shapes
:�����������
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_74/MatMulMatMul$dropout_57/dropout/SelectV2:output:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp,^lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp+^lstm_92/lstm_cell_96/MatMul/ReadVariableOp-^lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp^lstm_92/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2Z
+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp2X
*lstm_92/lstm_cell_96/MatMul/ReadVariableOp*lstm_92/lstm_cell_96/MatMul/ReadVariableOp2\
,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp2
lstm_92/whilelstm_92/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_dropout_57_layer_call_fn_22697339

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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696022a
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
�\
�
$__inference__traced_restore_22697637
file_prefix3
 assignvariableop_dense_74_kernel:	�.
 assignvariableop_1_dense_74_bias:A
.assignvariableop_2_lstm_92_lstm_cell_96_kernel:	�L
8assignvariableop_3_lstm_92_lstm_cell_96_recurrent_kernel:
��;
,assignvariableop_4_lstm_92_lstm_cell_96_bias:	�&
assignvariableop_5_iteration:	 *
 assignvariableop_6_learning_rate: H
5assignvariableop_7_adam_m_lstm_92_lstm_cell_96_kernel:	�H
5assignvariableop_8_adam_v_lstm_92_lstm_cell_96_kernel:	�S
?assignvariableop_9_adam_m_lstm_92_lstm_cell_96_recurrent_kernel:
��T
@assignvariableop_10_adam_v_lstm_92_lstm_cell_96_recurrent_kernel:
��C
4assignvariableop_11_adam_m_lstm_92_lstm_cell_96_bias:	�C
4assignvariableop_12_adam_v_lstm_92_lstm_cell_96_bias:	�=
*assignvariableop_13_adam_m_dense_74_kernel:	�=
*assignvariableop_14_adam_v_dense_74_kernel:	�6
(assignvariableop_15_adam_m_dense_74_bias:6
(assignvariableop_16_adam_v_dense_74_bias:%
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
AssignVariableOpAssignVariableOp assignvariableop_dense_74_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_74_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_92_lstm_cell_96_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_92_lstm_cell_96_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_92_lstm_cell_96_biasIdentity_4:output:0"/device:CPU:0*&
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
AssignVariableOp_7AssignVariableOp5assignvariableop_7_adam_m_lstm_92_lstm_cell_96_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp5assignvariableop_8_adam_v_lstm_92_lstm_cell_96_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp?assignvariableop_9_adam_m_lstm_92_lstm_cell_96_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp@assignvariableop_10_adam_v_lstm_92_lstm_cell_96_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_m_lstm_92_lstm_cell_96_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_v_lstm_92_lstm_cell_96_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_m_dense_74_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_v_dense_74_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_m_dense_74_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_v_dense_74_biasIdentity_16:output:0"/device:CPU:0*&
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
�
�
while_cond_22697103
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22697103___redundant_placeholder06
2while_while_cond_22697103___redundant_placeholder16
2while_while_cond_22697103___redundant_placeholder26
2while_while_cond_22697103___redundant_placeholder3
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
while_body_22696959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_96_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_96_matmul_readvariableop_resource:	�G
3while_lstm_cell_96_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_96_biasadd_readvariableop_resource:	���)while/lstm_cell_96/BiasAdd/ReadVariableOp�(while/lstm_cell_96/MatMul/ReadVariableOp�*while/lstm_cell_96/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_96/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_96/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/addAddV2#while/lstm_cell_96/MatMul:product:0%while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_96/BiasAddBiasAddwhile/lstm_cell_96/add:z:01while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_96/splitSplit+while/lstm_cell_96/split/split_dim:output:0#while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_96/SigmoidSigmoid!while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_1Sigmoid!while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mulMul while/lstm_cell_96/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_96/ReluRelu!while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_1Mulwhile/lstm_cell_96/Sigmoid:y:0%while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/add_1AddV2while/lstm_cell_96/mul:z:0while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_2Sigmoid!while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_96/Relu_1Reluwhile/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_2Mul while/lstm_cell_96/Sigmoid_2:y:0'while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_96/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_96/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_96/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_96/BiasAdd/ReadVariableOp)^while/lstm_cell_96/MatMul/ReadVariableOp+^while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_96_biasadd_readvariableop_resource4while_lstm_cell_96_biasadd_readvariableop_resource_0"l
3while_lstm_cell_96_matmul_1_readvariableop_resource5while_lstm_cell_96_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_96_matmul_readvariableop_resource3while_lstm_cell_96_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_96/BiasAdd/ReadVariableOp)while/lstm_cell_96/BiasAdd/ReadVariableOp2T
(while/lstm_cell_96/MatMul/ReadVariableOp(while/lstm_cell_96/MatMul/ReadVariableOp2X
*while/lstm_cell_96/MatMul_1/ReadVariableOp*while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
while_cond_22696958
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22696958___redundant_placeholder06
2while_while_cond_22696958___redundant_placeholder16
2while_while_cond_22696958___redundant_placeholder26
2while_while_cond_22696958___redundant_placeholder3
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
�
*__inference_lstm_92_layer_call_fn_22696721
inputs_0
unknown:	�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22695656p
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
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
while_cond_22695778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22695778___redundant_placeholder06
2while_while_cond_22695778___redundant_placeholder16
2while_while_cond_22695778___redundant_placeholder26
2while_while_cond_22695778___redundant_placeholder3
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

lstm_92_while_body_22696611,
(lstm_92_while_lstm_92_while_loop_counter2
.lstm_92_while_lstm_92_while_maximum_iterations
lstm_92_while_placeholder
lstm_92_while_placeholder_1
lstm_92_while_placeholder_2
lstm_92_while_placeholder_3+
'lstm_92_while_lstm_92_strided_slice_1_0g
clstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0:	�Q
=lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��K
<lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
lstm_92_while_identity
lstm_92_while_identity_1
lstm_92_while_identity_2
lstm_92_while_identity_3
lstm_92_while_identity_4
lstm_92_while_identity_5)
%lstm_92_while_lstm_92_strided_slice_1e
alstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensorL
9lstm_92_while_lstm_cell_96_matmul_readvariableop_resource:	�O
;lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource:
��I
:lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource:	���1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp�0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp�2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp�
?lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensor_0lstm_92_while_placeholderHlstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp;lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_92/while/lstm_cell_96/MatMulMatMul8lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp=lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
#lstm_92/while/lstm_cell_96/MatMul_1MatMullstm_92_while_placeholder_2:lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_92/while/lstm_cell_96/addAddV2+lstm_92/while/lstm_cell_96/MatMul:product:0-lstm_92/while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp<lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_92/while/lstm_cell_96/BiasAddBiasAdd"lstm_92/while/lstm_cell_96/add:z:09lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_92/while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_92/while/lstm_cell_96/splitSplit3lstm_92/while/lstm_cell_96/split/split_dim:output:0+lstm_92/while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
"lstm_92/while/lstm_cell_96/SigmoidSigmoid)lstm_92/while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:�����������
$lstm_92/while/lstm_cell_96/Sigmoid_1Sigmoid)lstm_92/while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
lstm_92/while/lstm_cell_96/mulMul(lstm_92/while/lstm_cell_96/Sigmoid_1:y:0lstm_92_while_placeholder_3*
T0*(
_output_shapes
:�����������
lstm_92/while/lstm_cell_96/ReluRelu)lstm_92/while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
 lstm_92/while/lstm_cell_96/mul_1Mul&lstm_92/while/lstm_cell_96/Sigmoid:y:0-lstm_92/while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
 lstm_92/while/lstm_cell_96/add_1AddV2"lstm_92/while/lstm_cell_96/mul:z:0$lstm_92/while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:�����������
$lstm_92/while/lstm_cell_96/Sigmoid_2Sigmoid)lstm_92/while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:�����������
!lstm_92/while/lstm_cell_96/Relu_1Relu$lstm_92/while/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
 lstm_92/while/lstm_cell_96/mul_2Mul(lstm_92/while/lstm_cell_96/Sigmoid_2:y:0/lstm_92/while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������z
8lstm_92/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_92_while_placeholder_1Alstm_92/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_92/while/lstm_cell_96/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_92/while/addAddV2lstm_92_while_placeholderlstm_92/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_92/while/add_1AddV2(lstm_92_while_lstm_92_while_loop_counterlstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_92/while/IdentityIdentitylstm_92/while/add_1:z:0^lstm_92/while/NoOp*
T0*
_output_shapes
: �
lstm_92/while/Identity_1Identity.lstm_92_while_lstm_92_while_maximum_iterations^lstm_92/while/NoOp*
T0*
_output_shapes
: q
lstm_92/while/Identity_2Identitylstm_92/while/add:z:0^lstm_92/while/NoOp*
T0*
_output_shapes
: �
lstm_92/while/Identity_3IdentityBlstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_92/while/NoOp*
T0*
_output_shapes
: �
lstm_92/while/Identity_4Identity$lstm_92/while/lstm_cell_96/mul_2:z:0^lstm_92/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_92/while/Identity_5Identity$lstm_92/while/lstm_cell_96/add_1:z:0^lstm_92/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_92/while/NoOpNoOp2^lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp1^lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp3^lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_92_while_identitylstm_92/while/Identity:output:0"=
lstm_92_while_identity_1!lstm_92/while/Identity_1:output:0"=
lstm_92_while_identity_2!lstm_92/while/Identity_2:output:0"=
lstm_92_while_identity_3!lstm_92/while/Identity_3:output:0"=
lstm_92_while_identity_4!lstm_92/while/Identity_4:output:0"=
lstm_92_while_identity_5!lstm_92/while/Identity_5:output:0"P
%lstm_92_while_lstm_92_strided_slice_1'lstm_92_while_lstm_92_strided_slice_1_0"z
:lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource<lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0"|
;lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource=lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0"x
9lstm_92_while_lstm_cell_96_matmul_readvariableop_resource;lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0"�
alstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensorclstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2f
1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp2d
0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp2h
2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697349

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
�
�
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696041

inputs#
lstm_92_22696010:	�$
lstm_92_22696012:
��
lstm_92_22696014:	�$
dense_74_22696035:	�
dense_74_22696037:
identity�� dense_74/StatefulPartitionedCall�lstm_92/StatefulPartitionedCall�
lstm_92/StatefulPartitionedCallStatefulPartitionedCallinputslstm_92_22696010lstm_92_22696012lstm_92_22696014*
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696009�
dropout_57/PartitionedCallPartitionedCall(lstm_92/StatefulPartitionedCall:output:0*
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696022�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall#dropout_57/PartitionedCall:output:0dense_74_22696035dense_74_22696037*
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
F__inference_dense_74_layer_call_and_return_conditional_losses_22696034x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_74/StatefulPartitionedCall ^lstm_92/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2B
lstm_92/StatefulPartitionedCalllstm_92/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696245

inputs>
+lstm_cell_96_matmul_readvariableop_resource:	�A
-lstm_cell_96_matmul_1_readvariableop_resource:
��;
,lstm_cell_96_biasadd_readvariableop_resource:	�
identity��#lstm_cell_96/BiasAdd/ReadVariableOp�"lstm_cell_96/MatMul/ReadVariableOp�$lstm_cell_96/MatMul_1/ReadVariableOp�while;
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
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
"lstm_cell_96/MatMul/ReadVariableOpReadVariableOp+lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_96/MatMulMatMulstrided_slice_2:output:0*lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_96/MatMul_1MatMulzeros:output:0,lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/addAddV2lstm_cell_96/MatMul:product:0lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_96/BiasAddBiasAddlstm_cell_96/add:z:0+lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_96/splitSplit%lstm_cell_96/split/split_dim:output:0lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_96/SigmoidSigmoidlstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_1Sigmoidlstm_cell_96/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_96/mulMullstm_cell_96/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_96/ReluRelulstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_1Mullstm_cell_96/Sigmoid:y:0lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_96/add_1AddV2lstm_cell_96/mul:z:0lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_2Sigmoidlstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_96/Relu_1Relulstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_2Mullstm_cell_96/Sigmoid_2:y:0!lstm_cell_96/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_96_matmul_readvariableop_resource-lstm_cell_96_matmul_1_readvariableop_resource,lstm_cell_96_biasadd_readvariableop_resource*
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
while_body_22696160*
condR
while_cond_22696159*M
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
NoOpNoOp$^lstm_cell_96/BiasAdd/ReadVariableOp#^lstm_cell_96/MatMul/ReadVariableOp%^lstm_cell_96/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_96/BiasAdd/ReadVariableOp#lstm_cell_96/BiasAdd/ReadVariableOp2H
"lstm_cell_96/MatMul/ReadVariableOp"lstm_cell_96/MatMul/ReadVariableOp2L
$lstm_cell_96/MatMul_1/ReadVariableOp$lstm_cell_96/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_sequential_76_layer_call_fn_22696316
lstm_92_input
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_92_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696288o
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
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_92_input
�
�
+__inference_dense_74_layer_call_fn_22697370

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
F__inference_dense_74_layer_call_and_return_conditional_losses_22696034o
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
�
�
*__inference_lstm_92_layer_call_fn_22696732
inputs_0
unknown:	�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22695849p
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
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�K
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696899
inputs_0>
+lstm_cell_96_matmul_readvariableop_resource:	�A
-lstm_cell_96_matmul_1_readvariableop_resource:
��;
,lstm_cell_96_biasadd_readvariableop_resource:	�
identity��#lstm_cell_96/BiasAdd/ReadVariableOp�"lstm_cell_96/MatMul/ReadVariableOp�$lstm_cell_96/MatMul_1/ReadVariableOp�while=
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
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
"lstm_cell_96/MatMul/ReadVariableOpReadVariableOp+lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_96/MatMulMatMulstrided_slice_2:output:0*lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_96/MatMul_1MatMulzeros:output:0,lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/addAddV2lstm_cell_96/MatMul:product:0lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_96/BiasAddBiasAddlstm_cell_96/add:z:0+lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_96/splitSplit%lstm_cell_96/split/split_dim:output:0lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_96/SigmoidSigmoidlstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_1Sigmoidlstm_cell_96/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_96/mulMullstm_cell_96/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_96/ReluRelulstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_1Mullstm_cell_96/Sigmoid:y:0lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_96/add_1AddV2lstm_cell_96/mul:z:0lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_2Sigmoidlstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_96/Relu_1Relulstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_2Mullstm_cell_96/Sigmoid_2:y:0!lstm_cell_96/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_96_matmul_readvariableop_resource-lstm_cell_96_matmul_1_readvariableop_resource,lstm_cell_96_biasadd_readvariableop_resource*
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
while_body_22696814*
condR
while_cond_22696813*M
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
NoOpNoOp$^lstm_cell_96/BiasAdd/ReadVariableOp#^lstm_cell_96/MatMul/ReadVariableOp%^lstm_cell_96/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_96/BiasAdd/ReadVariableOp#lstm_cell_96/BiasAdd/ReadVariableOp2H
"lstm_cell_96/MatMul/ReadVariableOp"lstm_cell_96/MatMul/ReadVariableOp2L
$lstm_cell_96/MatMul_1/ReadVariableOp$lstm_cell_96/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�9
�
while_body_22697249
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_96_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_96_matmul_readvariableop_resource:	�G
3while_lstm_cell_96_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_96_biasadd_readvariableop_resource:	���)while/lstm_cell_96/BiasAdd/ReadVariableOp�(while/lstm_cell_96/MatMul/ReadVariableOp�*while/lstm_cell_96/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_96/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_96/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/addAddV2#while/lstm_cell_96/MatMul:product:0%while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_96/BiasAddBiasAddwhile/lstm_cell_96/add:z:01while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_96/splitSplit+while/lstm_cell_96/split/split_dim:output:0#while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_96/SigmoidSigmoid!while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_1Sigmoid!while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mulMul while/lstm_cell_96/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_96/ReluRelu!while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_1Mulwhile/lstm_cell_96/Sigmoid:y:0%while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/add_1AddV2while/lstm_cell_96/mul:z:0while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_2Sigmoid!while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_96/Relu_1Reluwhile/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_2Mul while/lstm_cell_96/Sigmoid_2:y:0'while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_96/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_96/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_96/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_96/BiasAdd/ReadVariableOp)^while/lstm_cell_96/MatMul/ReadVariableOp+^while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_96_biasadd_readvariableop_resource4while_lstm_cell_96_biasadd_readvariableop_resource_0"l
3while_lstm_cell_96_matmul_1_readvariableop_resource5while_lstm_cell_96_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_96_matmul_readvariableop_resource3while_lstm_cell_96_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_96/BiasAdd/ReadVariableOp)while/lstm_cell_96/BiasAdd/ReadVariableOp2T
(while/lstm_cell_96/MatMul/ReadVariableOp(while/lstm_cell_96/MatMul/ReadVariableOp2X
*while/lstm_cell_96/MatMul_1/ReadVariableOp*while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
�9
�
while_body_22696814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_96_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_96_matmul_readvariableop_resource:	�G
3while_lstm_cell_96_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_96_biasadd_readvariableop_resource:	���)while/lstm_cell_96/BiasAdd/ReadVariableOp�(while/lstm_cell_96/MatMul/ReadVariableOp�*while/lstm_cell_96/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_96/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_96/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/addAddV2#while/lstm_cell_96/MatMul:product:0%while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_96/BiasAddBiasAddwhile/lstm_cell_96/add:z:01while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_96/splitSplit+while/lstm_cell_96/split/split_dim:output:0#while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_96/SigmoidSigmoid!while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_1Sigmoid!while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mulMul while/lstm_cell_96/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_96/ReluRelu!while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_1Mulwhile/lstm_cell_96/Sigmoid:y:0%while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/add_1AddV2while/lstm_cell_96/mul:z:0while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_2Sigmoid!while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_96/Relu_1Reluwhile/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_2Mul while/lstm_cell_96/Sigmoid_2:y:0'while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_96/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_96/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_96/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_96/BiasAdd/ReadVariableOp)^while/lstm_cell_96/MatMul/ReadVariableOp+^while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_96_biasadd_readvariableop_resource4while_lstm_cell_96_biasadd_readvariableop_resource_0"l
3while_lstm_cell_96_matmul_1_readvariableop_resource5while_lstm_cell_96_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_96_matmul_readvariableop_resource3while_lstm_cell_96_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_96/BiasAdd/ReadVariableOp)while/lstm_cell_96/BiasAdd/ReadVariableOp2T
(while/lstm_cell_96/MatMul/ReadVariableOp(while/lstm_cell_96/MatMul/ReadVariableOp2X
*while/lstm_cell_96/MatMul_1/ReadVariableOp*while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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

g
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696084

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
�
�
while_cond_22697248
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22697248___redundant_placeholder06
2while_while_cond_22697248___redundant_placeholder16
2while_while_cond_22697248___redundant_placeholder26
2while_while_cond_22697248___redundant_placeholder3
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
�K
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697334

inputs>
+lstm_cell_96_matmul_readvariableop_resource:	�A
-lstm_cell_96_matmul_1_readvariableop_resource:
��;
,lstm_cell_96_biasadd_readvariableop_resource:	�
identity��#lstm_cell_96/BiasAdd/ReadVariableOp�"lstm_cell_96/MatMul/ReadVariableOp�$lstm_cell_96/MatMul_1/ReadVariableOp�while;
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
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
"lstm_cell_96/MatMul/ReadVariableOpReadVariableOp+lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_96/MatMulMatMulstrided_slice_2:output:0*lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_96/MatMul_1MatMulzeros:output:0,lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/addAddV2lstm_cell_96/MatMul:product:0lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_96/BiasAddBiasAddlstm_cell_96/add:z:0+lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_96/splitSplit%lstm_cell_96/split/split_dim:output:0lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_96/SigmoidSigmoidlstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_1Sigmoidlstm_cell_96/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_96/mulMullstm_cell_96/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_96/ReluRelulstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_1Mullstm_cell_96/Sigmoid:y:0lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_96/add_1AddV2lstm_cell_96/mul:z:0lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_2Sigmoidlstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_96/Relu_1Relulstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_2Mullstm_cell_96/Sigmoid_2:y:0!lstm_cell_96/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_96_matmul_readvariableop_resource-lstm_cell_96_matmul_1_readvariableop_resource,lstm_cell_96_biasadd_readvariableop_resource*
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
while_body_22697249*
condR
while_cond_22697248*M
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
NoOpNoOp$^lstm_cell_96/BiasAdd/ReadVariableOp#^lstm_cell_96/MatMul/ReadVariableOp%^lstm_cell_96/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_96/BiasAdd/ReadVariableOp#lstm_cell_96/BiasAdd/ReadVariableOp2H
"lstm_cell_96/MatMul/ReadVariableOp"lstm_cell_96/MatMul/ReadVariableOp2L
$lstm_cell_96/MatMul_1/ReadVariableOp$lstm_cell_96/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_22696369
lstm_92_input
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_92_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
#__inference__wrapped_model_22695504o
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
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_92_input
�
f
-__inference_dropout_57_layer_call_fn_22697344

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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696084p
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
�
�
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697446

inputs
states_0
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
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
�
�
*__inference_lstm_92_layer_call_fn_22696754

inputs
unknown:	�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696245p
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
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
while_body_22695779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_96_22695803_0:	�1
while_lstm_cell_96_22695805_0:
��,
while_lstm_cell_96_22695807_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_96_22695803:	�/
while_lstm_cell_96_22695805:
��*
while_lstm_cell_96_22695807:	���*while/lstm_cell_96/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_96/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_96_22695803_0while_lstm_cell_96_22695805_0while_lstm_cell_96_22695807_0*
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695719r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_96/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_96/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:�����������
while/Identity_5Identity3while/lstm_cell_96/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:����������y

while/NoOpNoOp+^while/lstm_cell_96/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_96_22695803while_lstm_cell_96_22695803_0"<
while_lstm_cell_96_22695805while_lstm_cell_96_22695805_0"<
while_lstm_cell_96_22695807while_lstm_cell_96_22695807_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2X
*while/lstm_cell_96/StatefulPartitionedCall*while/lstm_cell_96/StatefulPartitionedCall: 
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
�
�
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696333
lstm_92_input#
lstm_92_22696319:	�$
lstm_92_22696321:
��
lstm_92_22696323:	�$
dense_74_22696327:	�
dense_74_22696329:
identity�� dense_74/StatefulPartitionedCall�lstm_92/StatefulPartitionedCall�
lstm_92/StatefulPartitionedCallStatefulPartitionedCalllstm_92_inputlstm_92_22696319lstm_92_22696321lstm_92_22696323*
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696009�
dropout_57/PartitionedCallPartitionedCall(lstm_92/StatefulPartitionedCall:output:0*
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696022�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall#dropout_57/PartitionedCall:output:0dense_74_22696327dense_74_22696329*
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
F__inference_dense_74_layer_call_and_return_conditional_losses_22696034x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_74/StatefulPartitionedCall ^lstm_92/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2B
lstm_92/StatefulPartitionedCalllstm_92/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_92_input
�
f
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696022

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
�
�
/__inference_lstm_cell_96_layer_call_fn_22697414

inputs
states_0
states_1
unknown:	�
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695719p
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
A:���������:����������:����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
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
�K
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697189

inputs>
+lstm_cell_96_matmul_readvariableop_resource:	�A
-lstm_cell_96_matmul_1_readvariableop_resource:
��;
,lstm_cell_96_biasadd_readvariableop_resource:	�
identity��#lstm_cell_96/BiasAdd/ReadVariableOp�"lstm_cell_96/MatMul/ReadVariableOp�$lstm_cell_96/MatMul_1/ReadVariableOp�while;
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
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
"lstm_cell_96/MatMul/ReadVariableOpReadVariableOp+lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_96/MatMulMatMulstrided_slice_2:output:0*lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_96/MatMul_1MatMulzeros:output:0,lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/addAddV2lstm_cell_96/MatMul:product:0lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_96/BiasAddBiasAddlstm_cell_96/add:z:0+lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_96/splitSplit%lstm_cell_96/split/split_dim:output:0lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_96/SigmoidSigmoidlstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_1Sigmoidlstm_cell_96/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_96/mulMullstm_cell_96/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_96/ReluRelulstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_1Mullstm_cell_96/Sigmoid:y:0lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_96/add_1AddV2lstm_cell_96/mul:z:0lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_2Sigmoidlstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_96/Relu_1Relulstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_2Mullstm_cell_96/Sigmoid_2:y:0!lstm_cell_96/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_96_matmul_readvariableop_resource-lstm_cell_96_matmul_1_readvariableop_resource,lstm_cell_96_biasadd_readvariableop_resource*
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
while_body_22697104*
condR
while_cond_22697103*M
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
NoOpNoOp$^lstm_cell_96/BiasAdd/ReadVariableOp#^lstm_cell_96/MatMul/ReadVariableOp%^lstm_cell_96/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_96/BiasAdd/ReadVariableOp#lstm_cell_96/BiasAdd/ReadVariableOp2H
"lstm_cell_96/MatMul/ReadVariableOp"lstm_cell_96/MatMul/ReadVariableOp2L
$lstm_cell_96/MatMul_1/ReadVariableOp$lstm_cell_96/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
F__inference_dense_74_layer_call_and_return_conditional_losses_22697380

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
�
�
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696350
lstm_92_input#
lstm_92_22696336:	�$
lstm_92_22696338:
��
lstm_92_22696340:	�$
dense_74_22696344:	�
dense_74_22696346:
identity�� dense_74/StatefulPartitionedCall�"dropout_57/StatefulPartitionedCall�lstm_92/StatefulPartitionedCall�
lstm_92/StatefulPartitionedCallStatefulPartitionedCalllstm_92_inputlstm_92_22696336lstm_92_22696338lstm_92_22696340*
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696245�
"dropout_57/StatefulPartitionedCallStatefulPartitionedCall(lstm_92/StatefulPartitionedCall:output:0*
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22696084�
 dense_74/StatefulPartitionedCallStatefulPartitionedCall+dropout_57/StatefulPartitionedCall:output:0dense_74_22696344dense_74_22696346*
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
F__inference_dense_74_layer_call_and_return_conditional_losses_22696034x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_74/StatefulPartitionedCall#^dropout_57/StatefulPartitionedCall ^lstm_92/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2H
"dropout_57/StatefulPartitionedCall"dropout_57/StatefulPartitionedCall2B
lstm_92/StatefulPartitionedCalllstm_92/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_92_input
�$
�
while_body_22695586
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_96_22695610_0:	�1
while_lstm_cell_96_22695612_0:
��,
while_lstm_cell_96_22695614_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_96_22695610:	�/
while_lstm_cell_96_22695612:
��*
while_lstm_cell_96_22695614:	���*while/lstm_cell_96/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
*while/lstm_cell_96/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_96_22695610_0while_lstm_cell_96_22695612_0while_lstm_cell_96_22695614_0*
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22695571r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_96/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_96/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:�����������
while/Identity_5Identity3while/lstm_cell_96/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:����������y

while/NoOpNoOp+^while/lstm_cell_96/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_96_22695610while_lstm_cell_96_22695610_0"<
while_lstm_cell_96_22695612while_lstm_cell_96_22695612_0"<
while_lstm_cell_96_22695614while_lstm_cell_96_22695614_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2X
*while/lstm_cell_96/StatefulPartitionedCall*while/lstm_cell_96/StatefulPartitionedCall: 
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
while_cond_22695923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22695923___redundant_placeholder06
2while_while_cond_22695923___redundant_placeholder16
2while_while_cond_22695923___redundant_placeholder26
2while_while_cond_22695923___redundant_placeholder3
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
�
0__inference_sequential_76_layer_call_fn_22696399

inputs
unknown:	�
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696288o
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
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�C
�

lstm_92_while_body_22696459,
(lstm_92_while_lstm_92_while_loop_counter2
.lstm_92_while_lstm_92_while_maximum_iterations
lstm_92_while_placeholder
lstm_92_while_placeholder_1
lstm_92_while_placeholder_2
lstm_92_while_placeholder_3+
'lstm_92_while_lstm_92_strided_slice_1_0g
clstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0:	�Q
=lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��K
<lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
lstm_92_while_identity
lstm_92_while_identity_1
lstm_92_while_identity_2
lstm_92_while_identity_3
lstm_92_while_identity_4
lstm_92_while_identity_5)
%lstm_92_while_lstm_92_strided_slice_1e
alstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensorL
9lstm_92_while_lstm_cell_96_matmul_readvariableop_resource:	�O
;lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource:
��I
:lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource:	���1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp�0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp�2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp�
?lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensor_0lstm_92_while_placeholderHlstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp;lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
!lstm_92/while/lstm_cell_96/MatMulMatMul8lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp=lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
#lstm_92/while/lstm_cell_96/MatMul_1MatMullstm_92_while_placeholder_2:lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_92/while/lstm_cell_96/addAddV2+lstm_92/while/lstm_cell_96/MatMul:product:0-lstm_92/while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp<lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
"lstm_92/while/lstm_cell_96/BiasAddBiasAdd"lstm_92/while/lstm_cell_96/add:z:09lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������l
*lstm_92/while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 lstm_92/while/lstm_cell_96/splitSplit3lstm_92/while/lstm_cell_96/split/split_dim:output:0+lstm_92/while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
"lstm_92/while/lstm_cell_96/SigmoidSigmoid)lstm_92/while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:�����������
$lstm_92/while/lstm_cell_96/Sigmoid_1Sigmoid)lstm_92/while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
lstm_92/while/lstm_cell_96/mulMul(lstm_92/while/lstm_cell_96/Sigmoid_1:y:0lstm_92_while_placeholder_3*
T0*(
_output_shapes
:�����������
lstm_92/while/lstm_cell_96/ReluRelu)lstm_92/while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
 lstm_92/while/lstm_cell_96/mul_1Mul&lstm_92/while/lstm_cell_96/Sigmoid:y:0-lstm_92/while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
 lstm_92/while/lstm_cell_96/add_1AddV2"lstm_92/while/lstm_cell_96/mul:z:0$lstm_92/while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:�����������
$lstm_92/while/lstm_cell_96/Sigmoid_2Sigmoid)lstm_92/while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:�����������
!lstm_92/while/lstm_cell_96/Relu_1Relu$lstm_92/while/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
 lstm_92/while/lstm_cell_96/mul_2Mul(lstm_92/while/lstm_cell_96/Sigmoid_2:y:0/lstm_92/while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������z
8lstm_92/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
2lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_92_while_placeholder_1Alstm_92/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_92/while/lstm_cell_96/mul_2:z:0*
_output_shapes
: *
element_dtype0:���U
lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_92/while/addAddV2lstm_92_while_placeholderlstm_92/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_92/while/add_1AddV2(lstm_92_while_lstm_92_while_loop_counterlstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_92/while/IdentityIdentitylstm_92/while/add_1:z:0^lstm_92/while/NoOp*
T0*
_output_shapes
: �
lstm_92/while/Identity_1Identity.lstm_92_while_lstm_92_while_maximum_iterations^lstm_92/while/NoOp*
T0*
_output_shapes
: q
lstm_92/while/Identity_2Identitylstm_92/while/add:z:0^lstm_92/while/NoOp*
T0*
_output_shapes
: �
lstm_92/while/Identity_3IdentityBlstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_92/while/NoOp*
T0*
_output_shapes
: �
lstm_92/while/Identity_4Identity$lstm_92/while/lstm_cell_96/mul_2:z:0^lstm_92/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_92/while/Identity_5Identity$lstm_92/while/lstm_cell_96/add_1:z:0^lstm_92/while/NoOp*
T0*(
_output_shapes
:�����������
lstm_92/while/NoOpNoOp2^lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp1^lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp3^lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_92_while_identitylstm_92/while/Identity:output:0"=
lstm_92_while_identity_1!lstm_92/while/Identity_1:output:0"=
lstm_92_while_identity_2!lstm_92/while/Identity_2:output:0"=
lstm_92_while_identity_3!lstm_92/while/Identity_3:output:0"=
lstm_92_while_identity_4!lstm_92/while/Identity_4:output:0"=
lstm_92_while_identity_5!lstm_92/while/Identity_5:output:0"P
%lstm_92_while_lstm_92_strided_slice_1'lstm_92_while_lstm_92_strided_slice_1_0"z
:lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource<lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0"|
;lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource=lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0"x
9lstm_92_while_lstm_cell_96_matmul_readvariableop_resource;lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0"�
alstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensorclstm_92_while_tensorarrayv2read_tensorlistgetitem_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2f
1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp1lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp2d
0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp0lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp2h
2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp2lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
0__inference_sequential_76_layer_call_fn_22696054
lstm_92_input
unknown:	�
	unknown_0:
��
	unknown_1:	�
	unknown_2:	�
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_92_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696041o
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
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_92_input
�]
�
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696551

inputsF
3lstm_92_lstm_cell_96_matmul_readvariableop_resource:	�I
5lstm_92_lstm_cell_96_matmul_1_readvariableop_resource:
��C
4lstm_92_lstm_cell_96_biasadd_readvariableop_resource:	�:
'dense_74_matmul_readvariableop_resource:	�6
(dense_74_biasadd_readvariableop_resource:
identity��dense_74/BiasAdd/ReadVariableOp�dense_74/MatMul/ReadVariableOp�+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp�*lstm_92/lstm_cell_96/MatMul/ReadVariableOp�,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp�lstm_92/whileC
lstm_92/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_sliceStridedSlicelstm_92/Shape:output:0$lstm_92/strided_slice/stack:output:0&lstm_92/strided_slice/stack_1:output:0&lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_92/zeros/packedPacklstm_92/strided_slice:output:0lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_92/zerosFilllstm_92/zeros/packed:output:0lstm_92/zeros/Const:output:0*
T0*(
_output_shapes
:����������[
lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
lstm_92/zeros_1/packedPacklstm_92/strided_slice:output:0!lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_92/zeros_1Filllstm_92/zeros_1/packed:output:0lstm_92/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������k
lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_92/transpose	Transposeinputslstm_92/transpose/perm:output:0*
T0*+
_output_shapes
:���������T
lstm_92/Shape_1Shapelstm_92/transpose:y:0*
T0*
_output_shapes
:g
lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_slice_1StridedSlicelstm_92/Shape_1:output:0&lstm_92/strided_slice_1/stack:output:0(lstm_92/strided_slice_1/stack_1:output:0(lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_92/TensorArrayV2TensorListReserve,lstm_92/TensorArrayV2/element_shape:output:0 lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
=lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_92/transpose:y:0Flstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���g
lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_slice_2StridedSlicelstm_92/transpose:y:0&lstm_92/strided_slice_2/stack:output:0(lstm_92/strided_slice_2/stack_1:output:0(lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
*lstm_92/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3lstm_92_lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_92/lstm_cell_96/MatMulMatMul lstm_92/strided_slice_2:output:02lstm_92/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5lstm_92_lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_92/lstm_cell_96/MatMul_1MatMullstm_92/zeros:output:04lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/addAddV2%lstm_92/lstm_cell_96/MatMul:product:0'lstm_92/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4lstm_92_lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_92/lstm_cell_96/BiasAddBiasAddlstm_92/lstm_cell_96/add:z:03lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������f
$lstm_92/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_92/lstm_cell_96/splitSplit-lstm_92/lstm_cell_96/split/split_dim:output:0%lstm_92/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split
lstm_92/lstm_cell_96/SigmoidSigmoid#lstm_92/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/Sigmoid_1Sigmoid#lstm_92/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/mulMul"lstm_92/lstm_cell_96/Sigmoid_1:y:0lstm_92/zeros_1:output:0*
T0*(
_output_shapes
:����������y
lstm_92/lstm_cell_96/ReluRelu#lstm_92/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/mul_1Mul lstm_92/lstm_cell_96/Sigmoid:y:0'lstm_92/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/add_1AddV2lstm_92/lstm_cell_96/mul:z:0lstm_92/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/Sigmoid_2Sigmoid#lstm_92/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������v
lstm_92/lstm_cell_96/Relu_1Relulstm_92/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_92/lstm_cell_96/mul_2Mul"lstm_92/lstm_cell_96/Sigmoid_2:y:0)lstm_92/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������v
%lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   f
$lstm_92/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_92/TensorArrayV2_1TensorListReserve.lstm_92/TensorArrayV2_1/element_shape:output:0-lstm_92/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���N
lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_92/whileWhile#lstm_92/while/loop_counter:output:0)lstm_92/while/maximum_iterations:output:0lstm_92/time:output:0 lstm_92/TensorArrayV2_1:handle:0lstm_92/zeros:output:0lstm_92/zeros_1:output:0 lstm_92/strided_slice_1:output:0?lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_92_lstm_cell_96_matmul_readvariableop_resource5lstm_92_lstm_cell_96_matmul_1_readvariableop_resource4lstm_92_lstm_cell_96_biasadd_readvariableop_resource*
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
lstm_92_while_body_22696459*'
condR
lstm_92_while_cond_22696458*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
8lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
*lstm_92/TensorArrayV2Stack/TensorListStackTensorListStacklstm_92/while:output:3Alstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elementsp
lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_92/strided_slice_3StridedSlice3lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_92/strided_slice_3/stack:output:0(lstm_92/strided_slice_3/stack_1:output:0(lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_maskm
lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_92/transpose_1	Transpose3lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_92/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������c
lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
dropout_57/IdentityIdentity lstm_92/strided_slice_3:output:0*
T0*(
_output_shapes
:�����������
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_74/MatMulMatMuldropout_57/Identity:output:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp,^lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp+^lstm_92/lstm_cell_96/MatMul/ReadVariableOp-^lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp^lstm_92/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2Z
+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp+lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp2X
*lstm_92/lstm_cell_96/MatMul/ReadVariableOp*lstm_92/lstm_cell_96/MatMul/ReadVariableOp2\
,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp,lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp2
lstm_92/whilelstm_92/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�q
�
#__inference__wrapped_model_22695504
lstm_92_inputT
Asequential_76_lstm_92_lstm_cell_96_matmul_readvariableop_resource:	�W
Csequential_76_lstm_92_lstm_cell_96_matmul_1_readvariableop_resource:
��Q
Bsequential_76_lstm_92_lstm_cell_96_biasadd_readvariableop_resource:	�H
5sequential_76_dense_74_matmul_readvariableop_resource:	�D
6sequential_76_dense_74_biasadd_readvariableop_resource:
identity��-sequential_76/dense_74/BiasAdd/ReadVariableOp�,sequential_76/dense_74/MatMul/ReadVariableOp�9sequential_76/lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp�8sequential_76/lstm_92/lstm_cell_96/MatMul/ReadVariableOp�:sequential_76/lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp�sequential_76/lstm_92/whileX
sequential_76/lstm_92/ShapeShapelstm_92_input*
T0*
_output_shapes
:s
)sequential_76/lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_76/lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_76/lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#sequential_76/lstm_92/strided_sliceStridedSlice$sequential_76/lstm_92/Shape:output:02sequential_76/lstm_92/strided_slice/stack:output:04sequential_76/lstm_92/strided_slice/stack_1:output:04sequential_76/lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_76/lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
"sequential_76/lstm_92/zeros/packedPack,sequential_76/lstm_92/strided_slice:output:0-sequential_76/lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_76/lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_76/lstm_92/zerosFill+sequential_76/lstm_92/zeros/packed:output:0*sequential_76/lstm_92/zeros/Const:output:0*
T0*(
_output_shapes
:����������i
&sequential_76/lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :��
$sequential_76/lstm_92/zeros_1/packedPack,sequential_76/lstm_92/strided_slice:output:0/sequential_76/lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_76/lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_76/lstm_92/zeros_1Fill-sequential_76/lstm_92/zeros_1/packed:output:0,sequential_76/lstm_92/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������y
$sequential_76/lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
sequential_76/lstm_92/transpose	Transposelstm_92_input-sequential_76/lstm_92/transpose/perm:output:0*
T0*+
_output_shapes
:���������p
sequential_76/lstm_92/Shape_1Shape#sequential_76/lstm_92/transpose:y:0*
T0*
_output_shapes
:u
+sequential_76/lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_76/lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_76/lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_76/lstm_92/strided_slice_1StridedSlice&sequential_76/lstm_92/Shape_1:output:04sequential_76/lstm_92/strided_slice_1/stack:output:06sequential_76/lstm_92/strided_slice_1/stack_1:output:06sequential_76/lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_76/lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
#sequential_76/lstm_92/TensorArrayV2TensorListReserve:sequential_76/lstm_92/TensorArrayV2/element_shape:output:0.sequential_76/lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ksequential_76/lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
=sequential_76/lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_76/lstm_92/transpose:y:0Tsequential_76/lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���u
+sequential_76/lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_76/lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_76/lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_76/lstm_92/strided_slice_2StridedSlice#sequential_76/lstm_92/transpose:y:04sequential_76/lstm_92/strided_slice_2/stack:output:06sequential_76/lstm_92/strided_slice_2/stack_1:output:06sequential_76/lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
8sequential_76/lstm_92/lstm_cell_96/MatMul/ReadVariableOpReadVariableOpAsequential_76_lstm_92_lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
)sequential_76/lstm_92/lstm_cell_96/MatMulMatMul.sequential_76/lstm_92/strided_slice_2:output:0@sequential_76/lstm_92/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:sequential_76/lstm_92/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOpCsequential_76_lstm_92_lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
+sequential_76/lstm_92/lstm_cell_96/MatMul_1MatMul$sequential_76/lstm_92/zeros:output:0Bsequential_76/lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&sequential_76/lstm_92/lstm_cell_96/addAddV23sequential_76/lstm_92/lstm_cell_96/MatMul:product:05sequential_76/lstm_92/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
9sequential_76/lstm_92/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOpBsequential_76_lstm_92_lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*sequential_76/lstm_92/lstm_cell_96/BiasAddBiasAdd*sequential_76/lstm_92/lstm_cell_96/add:z:0Asequential_76/lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
2sequential_76/lstm_92/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(sequential_76/lstm_92/lstm_cell_96/splitSplit;sequential_76/lstm_92/lstm_cell_96/split/split_dim:output:03sequential_76/lstm_92/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
*sequential_76/lstm_92/lstm_cell_96/SigmoidSigmoid1sequential_76/lstm_92/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:�����������
,sequential_76/lstm_92/lstm_cell_96/Sigmoid_1Sigmoid1sequential_76/lstm_92/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
&sequential_76/lstm_92/lstm_cell_96/mulMul0sequential_76/lstm_92/lstm_cell_96/Sigmoid_1:y:0&sequential_76/lstm_92/zeros_1:output:0*
T0*(
_output_shapes
:�����������
'sequential_76/lstm_92/lstm_cell_96/ReluRelu1sequential_76/lstm_92/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
(sequential_76/lstm_92/lstm_cell_96/mul_1Mul.sequential_76/lstm_92/lstm_cell_96/Sigmoid:y:05sequential_76/lstm_92/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
(sequential_76/lstm_92/lstm_cell_96/add_1AddV2*sequential_76/lstm_92/lstm_cell_96/mul:z:0,sequential_76/lstm_92/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:�����������
,sequential_76/lstm_92/lstm_cell_96/Sigmoid_2Sigmoid1sequential_76/lstm_92/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:�����������
)sequential_76/lstm_92/lstm_cell_96/Relu_1Relu,sequential_76/lstm_92/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
(sequential_76/lstm_92/lstm_cell_96/mul_2Mul0sequential_76/lstm_92/lstm_cell_96/Sigmoid_2:y:07sequential_76/lstm_92/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:�����������
3sequential_76/lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   t
2sequential_76/lstm_92/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
%sequential_76/lstm_92/TensorArrayV2_1TensorListReserve<sequential_76/lstm_92/TensorArrayV2_1/element_shape:output:0;sequential_76/lstm_92/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���\
sequential_76/lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_76/lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������j
(sequential_76/lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_76/lstm_92/whileWhile1sequential_76/lstm_92/while/loop_counter:output:07sequential_76/lstm_92/while/maximum_iterations:output:0#sequential_76/lstm_92/time:output:0.sequential_76/lstm_92/TensorArrayV2_1:handle:0$sequential_76/lstm_92/zeros:output:0&sequential_76/lstm_92/zeros_1:output:0.sequential_76/lstm_92/strided_slice_1:output:0Msequential_76/lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_76_lstm_92_lstm_cell_96_matmul_readvariableop_resourceCsequential_76_lstm_92_lstm_cell_96_matmul_1_readvariableop_resourceBsequential_76_lstm_92_lstm_cell_96_biasadd_readvariableop_resource*
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
)sequential_76_lstm_92_while_body_22695412*5
cond-R+
)sequential_76_lstm_92_while_cond_22695411*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations �
Fsequential_76/lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
8sequential_76/lstm_92/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_76/lstm_92/while:output:3Osequential_76/lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype0*
num_elements~
+sequential_76/lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������w
-sequential_76/lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_76/lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%sequential_76/lstm_92/strided_slice_3StridedSliceAsequential_76/lstm_92/TensorArrayV2Stack/TensorListStack:tensor:04sequential_76/lstm_92/strided_slice_3/stack:output:06sequential_76/lstm_92/strided_slice_3/stack_1:output:06sequential_76/lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask{
&sequential_76/lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
!sequential_76/lstm_92/transpose_1	TransposeAsequential_76/lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_76/lstm_92/transpose_1/perm:output:0*
T0*,
_output_shapes
:����������q
sequential_76/lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
!sequential_76/dropout_57/IdentityIdentity.sequential_76/lstm_92/strided_slice_3:output:0*
T0*(
_output_shapes
:�����������
,sequential_76/dense_74/MatMul/ReadVariableOpReadVariableOp5sequential_76_dense_74_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_76/dense_74/MatMulMatMul*sequential_76/dropout_57/Identity:output:04sequential_76/dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_76/dense_74/BiasAdd/ReadVariableOpReadVariableOp6sequential_76_dense_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_76/dense_74/BiasAddBiasAdd'sequential_76/dense_74/MatMul:product:05sequential_76/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_76/dense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_76/dense_74/BiasAdd/ReadVariableOp-^sequential_76/dense_74/MatMul/ReadVariableOp:^sequential_76/lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp9^sequential_76/lstm_92/lstm_cell_96/MatMul/ReadVariableOp;^sequential_76/lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp^sequential_76/lstm_92/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2^
-sequential_76/dense_74/BiasAdd/ReadVariableOp-sequential_76/dense_74/BiasAdd/ReadVariableOp2\
,sequential_76/dense_74/MatMul/ReadVariableOp,sequential_76/dense_74/MatMul/ReadVariableOp2v
9sequential_76/lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp9sequential_76/lstm_92/lstm_cell_96/BiasAdd/ReadVariableOp2t
8sequential_76/lstm_92/lstm_cell_96/MatMul/ReadVariableOp8sequential_76/lstm_92/lstm_cell_96/MatMul/ReadVariableOp2x
:sequential_76/lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp:sequential_76/lstm_92/lstm_cell_96/MatMul_1/ReadVariableOp2:
sequential_76/lstm_92/whilesequential_76/lstm_92/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_92_input
�K
�
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697044
inputs_0>
+lstm_cell_96_matmul_readvariableop_resource:	�A
-lstm_cell_96_matmul_1_readvariableop_resource:
��;
,lstm_cell_96_biasadd_readvariableop_resource:	�
identity��#lstm_cell_96/BiasAdd/ReadVariableOp�"lstm_cell_96/MatMul/ReadVariableOp�$lstm_cell_96/MatMul_1/ReadVariableOp�while=
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
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
"lstm_cell_96/MatMul/ReadVariableOpReadVariableOp+lstm_cell_96_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
lstm_cell_96/MatMulMatMulstrided_slice_2:output:0*lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_96_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
lstm_cell_96/MatMul_1MatMulzeros:output:0,lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/addAddV2lstm_cell_96/MatMul:product:0lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
#lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_96_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
lstm_cell_96/BiasAddBiasAddlstm_cell_96/add:z:0+lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������^
lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_96/splitSplit%lstm_cell_96/split/split_dim:output:0lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_splito
lstm_cell_96/SigmoidSigmoidlstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_1Sigmoidlstm_cell_96/split:output:1*
T0*(
_output_shapes
:����������x
lstm_cell_96/mulMullstm_cell_96/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������i
lstm_cell_96/ReluRelulstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_1Mullstm_cell_96/Sigmoid:y:0lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:����������|
lstm_cell_96/add_1AddV2lstm_cell_96/mul:z:0lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������q
lstm_cell_96/Sigmoid_2Sigmoidlstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������f
lstm_cell_96/Relu_1Relulstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
lstm_cell_96/mul_2Mullstm_cell_96/Sigmoid_2:y:0!lstm_cell_96/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_96_matmul_readvariableop_resource-lstm_cell_96_matmul_1_readvariableop_resource,lstm_cell_96_biasadd_readvariableop_resource*
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
while_body_22696959*
condR
while_cond_22696958*M
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
NoOpNoOp$^lstm_cell_96/BiasAdd/ReadVariableOp#^lstm_cell_96/MatMul/ReadVariableOp%^lstm_cell_96/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_96/BiasAdd/ReadVariableOp#lstm_cell_96/BiasAdd/ReadVariableOp2H
"lstm_cell_96/MatMul/ReadVariableOp"lstm_cell_96/MatMul/ReadVariableOp2L
$lstm_cell_96/MatMul_1/ReadVariableOp$lstm_cell_96/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
while_cond_22696813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22696813___redundant_placeholder06
2while_while_cond_22696813___redundant_placeholder16
2while_while_cond_22696813___redundant_placeholder26
2while_while_cond_22696813___redundant_placeholder3
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
while_body_22697104
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_96_matmul_readvariableop_resource_0:	�I
5while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��C
4while_lstm_cell_96_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_96_matmul_readvariableop_resource:	�G
3while_lstm_cell_96_matmul_1_readvariableop_resource:
��A
2while_lstm_cell_96_biasadd_readvariableop_resource:	���)while/lstm_cell_96/BiasAdd/ReadVariableOp�(while/lstm_cell_96/MatMul/ReadVariableOp�*while/lstm_cell_96/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
(while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/lstm_cell_96/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
while/lstm_cell_96/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/addAddV2#while/lstm_cell_96/MatMul:product:0%while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
)while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
while/lstm_cell_96/BiasAddBiasAddwhile/lstm_cell_96/add:z:01while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
"while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_96/splitSplit+while/lstm_cell_96/split/split_dim:output:0#while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split{
while/lstm_cell_96/SigmoidSigmoid!while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_1Sigmoid!while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mulMul while/lstm_cell_96/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������u
while/lstm_cell_96/ReluRelu!while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_1Mulwhile/lstm_cell_96/Sigmoid:y:0%while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/add_1AddV2while/lstm_cell_96/mul:z:0while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:����������}
while/lstm_cell_96/Sigmoid_2Sigmoid!while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:����������r
while/lstm_cell_96/Relu_1Reluwhile/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
while/lstm_cell_96/mul_2Mul while/lstm_cell_96/Sigmoid_2:y:0'while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:����������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_96/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_96/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:����������z
while/Identity_5Identitywhile/lstm_cell_96/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:�����������

while/NoOpNoOp*^while/lstm_cell_96/BiasAdd/ReadVariableOp)^while/lstm_cell_96/MatMul/ReadVariableOp+^while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_96_biasadd_readvariableop_resource4while_lstm_cell_96_biasadd_readvariableop_resource_0"l
3while_lstm_cell_96_matmul_1_readvariableop_resource5while_lstm_cell_96_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_96_matmul_readvariableop_resource3while_lstm_cell_96_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2V
)while/lstm_cell_96/BiasAdd/ReadVariableOp)while/lstm_cell_96/BiasAdd/ReadVariableOp2T
(while/lstm_cell_96/MatMul/ReadVariableOp(while/lstm_cell_96/MatMul/ReadVariableOp2X
*while/lstm_cell_96/MatMul_1/ReadVariableOp*while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
�3
�	
!__inference__traced_save_22697564
file_prefix.
*savev2_dense_74_kernel_read_readvariableop,
(savev2_dense_74_bias_read_readvariableop:
6savev2_lstm_92_lstm_cell_96_kernel_read_readvariableopD
@savev2_lstm_92_lstm_cell_96_recurrent_kernel_read_readvariableop8
4savev2_lstm_92_lstm_cell_96_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_lstm_92_lstm_cell_96_kernel_read_readvariableopA
=savev2_adam_v_lstm_92_lstm_cell_96_kernel_read_readvariableopK
Gsavev2_adam_m_lstm_92_lstm_cell_96_recurrent_kernel_read_readvariableopK
Gsavev2_adam_v_lstm_92_lstm_cell_96_recurrent_kernel_read_readvariableop?
;savev2_adam_m_lstm_92_lstm_cell_96_bias_read_readvariableop?
;savev2_adam_v_lstm_92_lstm_cell_96_bias_read_readvariableop5
1savev2_adam_m_dense_74_kernel_read_readvariableop5
1savev2_adam_v_dense_74_kernel_read_readvariableop3
/savev2_adam_m_dense_74_bias_read_readvariableop3
/savev2_adam_v_dense_74_bias_read_readvariableop&
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

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_74_kernel_read_readvariableop(savev2_dense_74_bias_read_readvariableop6savev2_lstm_92_lstm_cell_96_kernel_read_readvariableop@savev2_lstm_92_lstm_cell_96_recurrent_kernel_read_readvariableop4savev2_lstm_92_lstm_cell_96_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_lstm_92_lstm_cell_96_kernel_read_readvariableop=savev2_adam_v_lstm_92_lstm_cell_96_kernel_read_readvariableopGsavev2_adam_m_lstm_92_lstm_cell_96_recurrent_kernel_read_readvariableopGsavev2_adam_v_lstm_92_lstm_cell_96_recurrent_kernel_read_readvariableop;savev2_adam_m_lstm_92_lstm_cell_96_bias_read_readvariableop;savev2_adam_v_lstm_92_lstm_cell_96_bias_read_readvariableop1savev2_adam_m_dense_74_kernel_read_readvariableop1savev2_adam_v_dense_74_kernel_read_readvariableop/savev2_adam_m_dense_74_bias_read_readvariableop/savev2_adam_v_dense_74_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
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
�: :	�::	�:
��:�: : :	�:	�:
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
:	�:&"
 
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
:	�:%	!

_output_shapes
:	�:&
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
�
�
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697478

inputs
states_0
states_11
matmul_readvariableop_resource:	�4
 matmul_1_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
A:���������:����������:����������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
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
�
0__inference_sequential_76_layer_call_fn_22696384

inputs
unknown:	�
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696041o
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
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�S
�
)sequential_76_lstm_92_while_body_22695412H
Dsequential_76_lstm_92_while_sequential_76_lstm_92_while_loop_counterN
Jsequential_76_lstm_92_while_sequential_76_lstm_92_while_maximum_iterations+
'sequential_76_lstm_92_while_placeholder-
)sequential_76_lstm_92_while_placeholder_1-
)sequential_76_lstm_92_while_placeholder_2-
)sequential_76_lstm_92_while_placeholder_3G
Csequential_76_lstm_92_while_sequential_76_lstm_92_strided_slice_1_0�
sequential_76_lstm_92_while_tensorarrayv2read_tensorlistgetitem_sequential_76_lstm_92_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_76_lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0:	�_
Ksequential_76_lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0:
��Y
Jsequential_76_lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0:	�(
$sequential_76_lstm_92_while_identity*
&sequential_76_lstm_92_while_identity_1*
&sequential_76_lstm_92_while_identity_2*
&sequential_76_lstm_92_while_identity_3*
&sequential_76_lstm_92_while_identity_4*
&sequential_76_lstm_92_while_identity_5E
Asequential_76_lstm_92_while_sequential_76_lstm_92_strided_slice_1�
}sequential_76_lstm_92_while_tensorarrayv2read_tensorlistgetitem_sequential_76_lstm_92_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_76_lstm_92_while_lstm_cell_96_matmul_readvariableop_resource:	�]
Isequential_76_lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource:
��W
Hsequential_76_lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource:	���?sequential_76/lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp�>sequential_76/lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp�@sequential_76/lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp�
Msequential_76/lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
?sequential_76/lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_76_lstm_92_while_tensorarrayv2read_tensorlistgetitem_sequential_76_lstm_92_tensorarrayunstack_tensorlistfromtensor_0'sequential_76_lstm_92_while_placeholderVsequential_76/lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
>sequential_76/lstm_92/while/lstm_cell_96/MatMul/ReadVariableOpReadVariableOpIsequential_76_lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
/sequential_76/lstm_92/while/lstm_cell_96/MatMulMatMulFsequential_76/lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_76/lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
@sequential_76/lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOpReadVariableOpKsequential_76_lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
1sequential_76/lstm_92/while/lstm_cell_96/MatMul_1MatMul)sequential_76_lstm_92_while_placeholder_2Hsequential_76/lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_76/lstm_92/while/lstm_cell_96/addAddV29sequential_76/lstm_92/while/lstm_cell_96/MatMul:product:0;sequential_76/lstm_92/while/lstm_cell_96/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
?sequential_76/lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOpReadVariableOpJsequential_76_lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
0sequential_76/lstm_92/while/lstm_cell_96/BiasAddBiasAdd0sequential_76/lstm_92/while/lstm_cell_96/add:z:0Gsequential_76/lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������z
8sequential_76/lstm_92/while/lstm_cell_96/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
.sequential_76/lstm_92/while/lstm_cell_96/splitSplitAsequential_76/lstm_92/while/lstm_cell_96/split/split_dim:output:09sequential_76/lstm_92/while/lstm_cell_96/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split�
0sequential_76/lstm_92/while/lstm_cell_96/SigmoidSigmoid7sequential_76/lstm_92/while/lstm_cell_96/split:output:0*
T0*(
_output_shapes
:�����������
2sequential_76/lstm_92/while/lstm_cell_96/Sigmoid_1Sigmoid7sequential_76/lstm_92/while/lstm_cell_96/split:output:1*
T0*(
_output_shapes
:�����������
,sequential_76/lstm_92/while/lstm_cell_96/mulMul6sequential_76/lstm_92/while/lstm_cell_96/Sigmoid_1:y:0)sequential_76_lstm_92_while_placeholder_3*
T0*(
_output_shapes
:�����������
-sequential_76/lstm_92/while/lstm_cell_96/ReluRelu7sequential_76/lstm_92/while/lstm_cell_96/split:output:2*
T0*(
_output_shapes
:�����������
.sequential_76/lstm_92/while/lstm_cell_96/mul_1Mul4sequential_76/lstm_92/while/lstm_cell_96/Sigmoid:y:0;sequential_76/lstm_92/while/lstm_cell_96/Relu:activations:0*
T0*(
_output_shapes
:�����������
.sequential_76/lstm_92/while/lstm_cell_96/add_1AddV20sequential_76/lstm_92/while/lstm_cell_96/mul:z:02sequential_76/lstm_92/while/lstm_cell_96/mul_1:z:0*
T0*(
_output_shapes
:�����������
2sequential_76/lstm_92/while/lstm_cell_96/Sigmoid_2Sigmoid7sequential_76/lstm_92/while/lstm_cell_96/split:output:3*
T0*(
_output_shapes
:�����������
/sequential_76/lstm_92/while/lstm_cell_96/Relu_1Relu2sequential_76/lstm_92/while/lstm_cell_96/add_1:z:0*
T0*(
_output_shapes
:�����������
.sequential_76/lstm_92/while/lstm_cell_96/mul_2Mul6sequential_76/lstm_92/while/lstm_cell_96/Sigmoid_2:y:0=sequential_76/lstm_92/while/lstm_cell_96/Relu_1:activations:0*
T0*(
_output_shapes
:�����������
Fsequential_76/lstm_92/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
@sequential_76/lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_76_lstm_92_while_placeholder_1Osequential_76/lstm_92/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_76/lstm_92/while/lstm_cell_96/mul_2:z:0*
_output_shapes
: *
element_dtype0:���c
!sequential_76/lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
sequential_76/lstm_92/while/addAddV2'sequential_76_lstm_92_while_placeholder*sequential_76/lstm_92/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_76/lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
!sequential_76/lstm_92/while/add_1AddV2Dsequential_76_lstm_92_while_sequential_76_lstm_92_while_loop_counter,sequential_76/lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: �
$sequential_76/lstm_92/while/IdentityIdentity%sequential_76/lstm_92/while/add_1:z:0!^sequential_76/lstm_92/while/NoOp*
T0*
_output_shapes
: �
&sequential_76/lstm_92/while/Identity_1IdentityJsequential_76_lstm_92_while_sequential_76_lstm_92_while_maximum_iterations!^sequential_76/lstm_92/while/NoOp*
T0*
_output_shapes
: �
&sequential_76/lstm_92/while/Identity_2Identity#sequential_76/lstm_92/while/add:z:0!^sequential_76/lstm_92/while/NoOp*
T0*
_output_shapes
: �
&sequential_76/lstm_92/while/Identity_3IdentityPsequential_76/lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_76/lstm_92/while/NoOp*
T0*
_output_shapes
: �
&sequential_76/lstm_92/while/Identity_4Identity2sequential_76/lstm_92/while/lstm_cell_96/mul_2:z:0!^sequential_76/lstm_92/while/NoOp*
T0*(
_output_shapes
:�����������
&sequential_76/lstm_92/while/Identity_5Identity2sequential_76/lstm_92/while/lstm_cell_96/add_1:z:0!^sequential_76/lstm_92/while/NoOp*
T0*(
_output_shapes
:�����������
 sequential_76/lstm_92/while/NoOpNoOp@^sequential_76/lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp?^sequential_76/lstm_92/while/lstm_cell_96/MatMul/ReadVariableOpA^sequential_76/lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_76_lstm_92_while_identity-sequential_76/lstm_92/while/Identity:output:0"Y
&sequential_76_lstm_92_while_identity_1/sequential_76/lstm_92/while/Identity_1:output:0"Y
&sequential_76_lstm_92_while_identity_2/sequential_76/lstm_92/while/Identity_2:output:0"Y
&sequential_76_lstm_92_while_identity_3/sequential_76/lstm_92/while/Identity_3:output:0"Y
&sequential_76_lstm_92_while_identity_4/sequential_76/lstm_92/while/Identity_4:output:0"Y
&sequential_76_lstm_92_while_identity_5/sequential_76/lstm_92/while/Identity_5:output:0"�
Hsequential_76_lstm_92_while_lstm_cell_96_biasadd_readvariableop_resourceJsequential_76_lstm_92_while_lstm_cell_96_biasadd_readvariableop_resource_0"�
Isequential_76_lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resourceKsequential_76_lstm_92_while_lstm_cell_96_matmul_1_readvariableop_resource_0"�
Gsequential_76_lstm_92_while_lstm_cell_96_matmul_readvariableop_resourceIsequential_76_lstm_92_while_lstm_cell_96_matmul_readvariableop_resource_0"�
Asequential_76_lstm_92_while_sequential_76_lstm_92_strided_slice_1Csequential_76_lstm_92_while_sequential_76_lstm_92_strided_slice_1_0"�
}sequential_76_lstm_92_while_tensorarrayv2read_tensorlistgetitem_sequential_76_lstm_92_tensorarrayunstack_tensorlistfromtensorsequential_76_lstm_92_while_tensorarrayv2read_tensorlistgetitem_sequential_76_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :����������:����������: : : : : 2�
?sequential_76/lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp?sequential_76/lstm_92/while/lstm_cell_96/BiasAdd/ReadVariableOp2�
>sequential_76/lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp>sequential_76/lstm_92/while/lstm_cell_96/MatMul/ReadVariableOp2�
@sequential_76/lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp@sequential_76/lstm_92/while/lstm_cell_96/MatMul_1/ReadVariableOp: 
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
while_cond_22695585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_22695585___redundant_placeholder06
2while_while_cond_22695585___redundant_placeholder16
2while_while_cond_22695585___redundant_placeholder26
2while_while_cond_22695585___redundant_placeholder3
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
�

g
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697361

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
�	
�
F__inference_dense_74_layer_call_and_return_conditional_losses_22696034

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
�
�
*__inference_lstm_92_layer_call_fn_22696743

inputs
unknown:	�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696009p
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
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
lstm_92_while_cond_22696610,
(lstm_92_while_lstm_92_while_loop_counter2
.lstm_92_while_lstm_92_while_maximum_iterations
lstm_92_while_placeholder
lstm_92_while_placeholder_1
lstm_92_while_placeholder_2
lstm_92_while_placeholder_3.
*lstm_92_while_less_lstm_92_strided_slice_1F
Blstm_92_while_lstm_92_while_cond_22696610___redundant_placeholder0F
Blstm_92_while_lstm_92_while_cond_22696610___redundant_placeholder1F
Blstm_92_while_lstm_92_while_cond_22696610___redundant_placeholder2F
Blstm_92_while_lstm_92_while_cond_22696610___redundant_placeholder3
lstm_92_while_identity
�
lstm_92/while/LessLesslstm_92_while_placeholder*lstm_92_while_less_lstm_92_strided_slice_1*
T0*
_output_shapes
: [
lstm_92/while/IdentityIdentitylstm_92/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_92_while_identitylstm_92/while/Identity:output:0*(
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
:"�
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
lstm_92_input:
serving_default_lstm_92_input:0���������<
dense_740
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
0__inference_sequential_76_layer_call_fn_22696054
0__inference_sequential_76_layer_call_fn_22696384
0__inference_sequential_76_layer_call_fn_22696399
0__inference_sequential_76_layer_call_fn_22696316�
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696551
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696710
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696333
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696350�
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
#__inference__wrapped_model_22695504lstm_92_input"�
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
*__inference_lstm_92_layer_call_fn_22696721
*__inference_lstm_92_layer_call_fn_22696732
*__inference_lstm_92_layer_call_fn_22696743
*__inference_lstm_92_layer_call_fn_22696754�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696899
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697044
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697189
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697334�
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
-__inference_dropout_57_layer_call_fn_22697339
-__inference_dropout_57_layer_call_fn_22697344�
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697349
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697361�
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
+__inference_dense_74_layer_call_fn_22697370�
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
F__inference_dense_74_layer_call_and_return_conditional_losses_22697380�
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
": 	�2dense_74/kernel
:2dense_74/bias
.:,	�2lstm_92/lstm_cell_96/kernel
9:7
��2%lstm_92/lstm_cell_96/recurrent_kernel
(:&�2lstm_92/lstm_cell_96/bias
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
0__inference_sequential_76_layer_call_fn_22696054lstm_92_input"�
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
0__inference_sequential_76_layer_call_fn_22696384inputs"�
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
0__inference_sequential_76_layer_call_fn_22696399inputs"�
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
0__inference_sequential_76_layer_call_fn_22696316lstm_92_input"�
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696551inputs"�
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696710inputs"�
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696333lstm_92_input"�
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696350lstm_92_input"�
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
&__inference_signature_wrapper_22696369lstm_92_input"�
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
*__inference_lstm_92_layer_call_fn_22696721inputs_0"�
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
*__inference_lstm_92_layer_call_fn_22696732inputs_0"�
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
*__inference_lstm_92_layer_call_fn_22696743inputs"�
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
*__inference_lstm_92_layer_call_fn_22696754inputs"�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696899inputs_0"�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697044inputs_0"�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697189inputs"�
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
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697334inputs"�
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
/__inference_lstm_cell_96_layer_call_fn_22697397
/__inference_lstm_cell_96_layer_call_fn_22697414�
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697446
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697478�
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
-__inference_dropout_57_layer_call_fn_22697339inputs"�
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
-__inference_dropout_57_layer_call_fn_22697344inputs"�
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697349inputs"�
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
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697361inputs"�
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
+__inference_dense_74_layer_call_fn_22697370inputs"�
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
F__inference_dense_74_layer_call_and_return_conditional_losses_22697380inputs"�
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
3:1	�2"Adam/m/lstm_92/lstm_cell_96/kernel
3:1	�2"Adam/v/lstm_92/lstm_cell_96/kernel
>:<
��2,Adam/m/lstm_92/lstm_cell_96/recurrent_kernel
>:<
��2,Adam/v/lstm_92/lstm_cell_96/recurrent_kernel
-:+�2 Adam/m/lstm_92/lstm_cell_96/bias
-:+�2 Adam/v/lstm_92/lstm_cell_96/bias
':%	�2Adam/m/dense_74/kernel
':%	�2Adam/v/dense_74/kernel
 :2Adam/m/dense_74/bias
 :2Adam/v/dense_74/bias
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
/__inference_lstm_cell_96_layer_call_fn_22697397inputsstates_0states_1"�
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
/__inference_lstm_cell_96_layer_call_fn_22697414inputsstates_0states_1"�
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697446inputsstates_0states_1"�
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697478inputsstates_0states_1"�
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
#__inference__wrapped_model_22695504x%&'#$:�7
0�-
+�(
lstm_92_input���������
� "3�0
.
dense_74"�
dense_74����������
F__inference_dense_74_layer_call_and_return_conditional_losses_22697380d#$0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_74_layer_call_fn_22697370Y#$0�-
&�#
!�
inputs����������
� "!�
unknown����������
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697349e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
H__inference_dropout_57_layer_call_and_return_conditional_losses_22697361e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
-__inference_dropout_57_layer_call_fn_22697339Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
-__inference_dropout_57_layer_call_fn_22697344Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
E__inference_lstm_92_layer_call_and_return_conditional_losses_22696899�%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "-�*
#� 
tensor_0����������
� �
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697044�%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "-�*
#� 
tensor_0����������
� �
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697189u%&'?�<
5�2
$�!
inputs���������

 
p 

 
� "-�*
#� 
tensor_0����������
� �
E__inference_lstm_92_layer_call_and_return_conditional_losses_22697334u%&'?�<
5�2
$�!
inputs���������

 
p

 
� "-�*
#� 
tensor_0����������
� �
*__inference_lstm_92_layer_call_fn_22696721z%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ""�
unknown�����������
*__inference_lstm_92_layer_call_fn_22696732z%&'O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ""�
unknown�����������
*__inference_lstm_92_layer_call_fn_22696743j%&'?�<
5�2
$�!
inputs���������

 
p 

 
� ""�
unknown�����������
*__inference_lstm_92_layer_call_fn_22696754j%&'?�<
5�2
$�!
inputs���������

 
p

 
� ""�
unknown�����������
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697446�%&'��
x�u
 �
inputs���������
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
J__inference_lstm_cell_96_layer_call_and_return_conditional_losses_22697478�%&'��
x�u
 �
inputs���������
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
/__inference_lstm_cell_96_layer_call_fn_22697397�%&'��
x�u
 �
inputs���������
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
/__inference_lstm_cell_96_layer_call_fn_22697414�%&'��
x�u
 �
inputs���������
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
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696333y%&'#$B�?
8�5
+�(
lstm_92_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696350y%&'#$B�?
8�5
+�(
lstm_92_input���������
p

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696551r%&'#$;�8
1�.
$�!
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_76_layer_call_and_return_conditional_losses_22696710r%&'#$;�8
1�.
$�!
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
0__inference_sequential_76_layer_call_fn_22696054n%&'#$B�?
8�5
+�(
lstm_92_input���������
p 

 
� "!�
unknown����������
0__inference_sequential_76_layer_call_fn_22696316n%&'#$B�?
8�5
+�(
lstm_92_input���������
p

 
� "!�
unknown����������
0__inference_sequential_76_layer_call_fn_22696384g%&'#$;�8
1�.
$�!
inputs���������
p 

 
� "!�
unknown����������
0__inference_sequential_76_layer_call_fn_22696399g%&'#$;�8
1�.
$�!
inputs���������
p

 
� "!�
unknown����������
&__inference_signature_wrapper_22696369�%&'#$K�H
� 
A�>
<
lstm_92_input+�(
lstm_92_input���������"3�0
.
dense_74"�
dense_74���������