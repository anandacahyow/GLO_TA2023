╩Ј
ђ¤
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
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
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
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
ѕ"serve*2.11.02v2.11.0-rc2-15-g6290819256d8ци
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
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
ђ
Adam/v/dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_30/bias
y
(Adam/v/dense_30/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_30/bias*
_output_shapes
:*
dtype0
ђ
Adam/m/dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_30/bias
y
(Adam/m/dense_30/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_30/bias*
_output_shapes
:*
dtype0
Ѕ
Adam/v/dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┤*'
shared_nameAdam/v/dense_30/kernel
ѓ
*Adam/v/dense_30/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_30/kernel*
_output_shapes
:	┤*
dtype0
Ѕ
Adam/m/dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┤*'
shared_nameAdam/m/dense_30/kernel
ѓ
*Adam/m/dense_30/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_30/kernel*
_output_shapes
:	┤*
dtype0
Ў
 Adam/v/lstm_23/lstm_cell_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:л*1
shared_name" Adam/v/lstm_23/lstm_cell_23/bias
њ
4Adam/v/lstm_23/lstm_cell_23/bias/Read/ReadVariableOpReadVariableOp Adam/v/lstm_23/lstm_cell_23/bias*
_output_shapes	
:л*
dtype0
Ў
 Adam/m/lstm_23/lstm_cell_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:л*1
shared_name" Adam/m/lstm_23/lstm_cell_23/bias
њ
4Adam/m/lstm_23/lstm_cell_23/bias/Read/ReadVariableOpReadVariableOp Adam/m/lstm_23/lstm_cell_23/bias*
_output_shapes	
:л*
dtype0
Х
,Adam/v/lstm_23/lstm_cell_23/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
┤л*=
shared_name.,Adam/v/lstm_23/lstm_cell_23/recurrent_kernel
»
@Adam/v/lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/v/lstm_23/lstm_cell_23/recurrent_kernel* 
_output_shapes
:
┤л*
dtype0
Х
,Adam/m/lstm_23/lstm_cell_23/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
┤л*=
shared_name.,Adam/m/lstm_23/lstm_cell_23/recurrent_kernel
»
@Adam/m/lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/m/lstm_23/lstm_cell_23/recurrent_kernel* 
_output_shapes
:
┤л*
dtype0
А
"Adam/v/lstm_23/lstm_cell_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	л*3
shared_name$"Adam/v/lstm_23/lstm_cell_23/kernel
џ
6Adam/v/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_23/lstm_cell_23/kernel*
_output_shapes
:	л*
dtype0
А
"Adam/m/lstm_23/lstm_cell_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	л*3
shared_name$"Adam/m/lstm_23/lstm_cell_23/kernel
џ
6Adam/m/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_23/lstm_cell_23/kernel*
_output_shapes
:	л*
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
І
lstm_23/lstm_cell_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:л**
shared_namelstm_23/lstm_cell_23/bias
ё
-lstm_23/lstm_cell_23/bias/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/bias*
_output_shapes	
:л*
dtype0
е
%lstm_23/lstm_cell_23/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
┤л*6
shared_name'%lstm_23/lstm_cell_23/recurrent_kernel
А
9lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_23/lstm_cell_23/recurrent_kernel* 
_output_shapes
:
┤л*
dtype0
Њ
lstm_23/lstm_cell_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	л*,
shared_namelstm_23/lstm_cell_23/kernel
ї
/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/kernel*
_output_shapes
:	л*
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:*
dtype0
{
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┤* 
shared_namedense_30/kernel
t
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes
:	┤*
dtype0
ѕ
serving_default_lstm_23_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
├
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_23_inputlstm_23/lstm_cell_23/kernel%lstm_23/lstm_cell_23/recurrent_kernellstm_23/lstm_cell_23/biasdense_30/kerneldense_30/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ */
f*R(
&__inference_signature_wrapper_21002227

NoOpNoOp
а2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*█1
valueЛ1B╬1 BК1
Д
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
┴
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
Ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
д
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
░
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
Ђ
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
Ъ

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
с
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
Љ
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
Њ
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
VARIABLE_VALUEdense_30/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_30/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_23/lstm_cell_23/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_23/lstm_cell_23/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_23/lstm_cell_23/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

c0
d1
e2*
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
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
f0
h1
j2
l3
n4*
'
g0
i1
k2
m3
o4*
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
Њ
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

utrace_0
vtrace_1* 

wtrace_0
xtrace_1* 
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
y	variables
z	keras_api
	{total
	|count*
J
}	variables
~	keras_api
	total

ђcount
Ђ
_fn_kwargs*
M
ѓ	variables
Ѓ	keras_api

ёtotal

Ёcount
є
_fn_kwargs*
mg
VARIABLE_VALUE"Adam/m/lstm_23/lstm_cell_23/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_23/lstm_cell_23/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/m/lstm_23/lstm_cell_23/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/v/lstm_23/lstm_cell_23/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_23/lstm_cell_23/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_23/lstm_cell_23/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_30/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_30/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_30/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_30/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
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
{0
|1*

y	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
ђ1*

}	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ё0
Ё1*

ѓ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ј

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOp9lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOp-lstm_23/lstm_cell_23/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOp6Adam/v/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOp@Adam/m/lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOp@Adam/v/lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOp4Adam/m/lstm_23/lstm_cell_23/bias/Read/ReadVariableOp4Adam/v/lstm_23/lstm_cell_23/bias/Read/ReadVariableOp*Adam/m/dense_30/kernel/Read/ReadVariableOp*Adam/v/dense_30/kernel/Read/ReadVariableOp(Adam/m/dense_30/bias/Read/ReadVariableOp(Adam/v/dense_30/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*$
Tin
2	*
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_save_21003428
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_30/kerneldense_30/biaslstm_23/lstm_cell_23/kernel%lstm_23/lstm_cell_23/recurrent_kernellstm_23/lstm_cell_23/bias	iterationlearning_rate"Adam/m/lstm_23/lstm_cell_23/kernel"Adam/v/lstm_23/lstm_cell_23/kernel,Adam/m/lstm_23/lstm_cell_23/recurrent_kernel,Adam/v/lstm_23/lstm_cell_23/recurrent_kernel Adam/m/lstm_23/lstm_cell_23/bias Adam/v/lstm_23/lstm_cell_23/biasAdam/m/dense_30/kernelAdam/v/dense_30/kernelAdam/m/dense_30/biasAdam/v/dense_30/biastotal_2count_2total_1count_1totalcount*#
Tin
2*
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
GPU 2J 8ѓ *-
f(R&
$__inference__traced_restore_21003507д└
├
═
while_cond_21001636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21001636___redundant_placeholder06
2while_while_cond_21001636___redundant_placeholder16
2while_while_cond_21001636___redundant_placeholder26
2while_while_cond_21001636___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
П
ш
0__inference_sequential_26_layer_call_fn_21002242

inputs
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
	unknown_2:	┤
	unknown_3:
identityѕбStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_sequential_26_layer_call_and_return_conditional_losses_21001899o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
═	
Э
F__inference_dense_30_layer_call_and_return_conditional_losses_21001892

inputs1
matmul_readvariableop_resource:	┤-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	┤*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ┤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
ц

ь
lstm_23_while_cond_21002468,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1F
Blstm_23_while_lstm_23_while_cond_21002468___redundant_placeholder0F
Blstm_23_while_lstm_23_while_cond_21002468___redundant_placeholder1F
Blstm_23_while_lstm_23_while_cond_21002468___redundant_placeholder2F
Blstm_23_while_lstm_23_while_cond_21002468___redundant_placeholder3
lstm_23_while_identity
ѓ
lstm_23/while/LessLesslstm_23_while_placeholder*lstm_23_while_less_lstm_23_strided_slice_1*
T0*
_output_shapes
: [
lstm_23/while/IdentityIdentitylstm_23/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_23_while_identitylstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
П
ш
0__inference_sequential_26_layer_call_fn_21002257

inputs
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
	unknown_2:	┤
	unknown_3:
identityѕбStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ьK
а
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002902
inputs_0>
+lstm_cell_23_matmul_readvariableop_resource:	лA
-lstm_cell_23_matmul_1_readvariableop_resource:
┤л;
,lstm_cell_23_biasadd_readvariableop_resource:	л
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile=
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0ќ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лћ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0љ
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лї
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЇ
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splito
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤x
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ┤i
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤|
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤f
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤І
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : і
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21002817*
condR
while_cond_21002816*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤└
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
├
═
while_cond_21002816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21002816___redundant_placeholder06
2while_while_cond_21002816___redundant_placeholder16
2while_while_cond_21002816___redundant_placeholder26
2while_while_cond_21002816___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
ќ
║
*__inference_lstm_23_layer_call_fn_21002590
inputs_0
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001707p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
└9
н
while_body_21001782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_23_matmul_readvariableop_resource_0:	лI
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лC
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_23_matmul_readvariableop_resource:	лG
3while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лA
2while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0║
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лб
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0А
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лъ
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЏ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0Д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split{
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ┤u
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ј
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤r
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤z
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
ќ
║
*__inference_lstm_23_layer_call_fn_21002579
inputs_0
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001514p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
«d
П
$__inference__traced_restore_21003507
file_prefix3
 assignvariableop_dense_30_kernel:	┤.
 assignvariableop_1_dense_30_bias:A
.assignvariableop_2_lstm_23_lstm_cell_23_kernel:	лL
8assignvariableop_3_lstm_23_lstm_cell_23_recurrent_kernel:
┤л;
,assignvariableop_4_lstm_23_lstm_cell_23_bias:	л&
assignvariableop_5_iteration:	 *
 assignvariableop_6_learning_rate: H
5assignvariableop_7_adam_m_lstm_23_lstm_cell_23_kernel:	лH
5assignvariableop_8_adam_v_lstm_23_lstm_cell_23_kernel:	лS
?assignvariableop_9_adam_m_lstm_23_lstm_cell_23_recurrent_kernel:
┤лT
@assignvariableop_10_adam_v_lstm_23_lstm_cell_23_recurrent_kernel:
┤лC
4assignvariableop_11_adam_m_lstm_23_lstm_cell_23_bias:	лC
4assignvariableop_12_adam_v_lstm_23_lstm_cell_23_bias:	л=
*assignvariableop_13_adam_m_dense_30_kernel:	┤=
*assignvariableop_14_adam_v_dense_30_kernel:	┤6
(assignvariableop_15_adam_m_dense_30_bias:6
(assignvariableop_16_adam_v_dense_30_bias:%
assignvariableop_17_total_2: %
assignvariableop_18_count_2: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: #
assignvariableop_21_total: #
assignvariableop_22_count: 
identity_24ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Ю

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├	
value╣	BХ	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHа
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ќ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_dense_30_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_30_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_23_lstm_cell_23_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_23_lstm_cell_23_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_23_lstm_cell_23_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_5AssignVariableOpassignvariableop_5_iterationIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_6AssignVariableOp assignvariableop_6_learning_rateIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_7AssignVariableOp5assignvariableop_7_adam_m_lstm_23_lstm_cell_23_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_8AssignVariableOp5assignvariableop_8_adam_v_lstm_23_lstm_cell_23_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_9AssignVariableOp?assignvariableop_9_adam_m_lstm_23_lstm_cell_23_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_10AssignVariableOp@assignvariableop_10_adam_v_lstm_23_lstm_cell_23_recurrent_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_m_lstm_23_lstm_cell_23_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_v_lstm_23_lstm_cell_23_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_m_dense_30_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_v_dense_30_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_m_dense_30_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_v_dense_30_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ╔
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: Х
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
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
■
И
*__inference_lstm_23_layer_call_fn_21002601

inputs
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001867p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╩K
ъ
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002103

inputs>
+lstm_cell_23_matmul_readvariableop_resource:	лA
-lstm_cell_23_matmul_1_readvariableop_resource:
┤л;
,lstm_cell_23_biasadd_readvariableop_resource:	л
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0ќ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лћ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0љ
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лї
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЇ
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splito
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤x
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ┤i
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤|
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤f
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤І
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : і
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21002018*
condR
while_cond_21002017*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤└
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╩K
ъ
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003192

inputs>
+lstm_cell_23_matmul_readvariableop_resource:	лA
-lstm_cell_23_matmul_1_readvariableop_resource:
┤л;
,lstm_cell_23_biasadd_readvariableop_resource:	л
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0ќ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лћ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0љ
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лї
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЇ
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splito
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤x
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ┤i
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤|
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤f
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤І
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : і
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21003107*
condR
while_cond_21003106*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤└
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
█e
Њ
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002568

inputsF
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:	лI
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:
┤лC
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	л:
'dense_30_matmul_readvariableop_resource:	┤6
(dense_30_biasadd_readvariableop_resource:
identityѕбdense_30/BiasAdd/ReadVariableOpбdense_30/MatMul/ReadVariableOpб+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpб*lstm_23/lstm_cell_23/MatMul/ReadVariableOpб,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpбlstm_23/whileC
lstm_23/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤І
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ё
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*(
_output_shapes
:         ┤[
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤Ј
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    І
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ┤k
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_23/transpose	Transposeinputslstm_23/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:g
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЪ
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0«
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лц
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0е
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лц
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЮ
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Г
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лf
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :щ
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤Ђ
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤љ
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*(
_output_shapes
:         ┤y
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ъ
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ћ
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤Ђ
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤v
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Б
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤v
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   f
$lstm_23/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_23/TensorArrayV2_1TensorListReserve.lstm_23/TensorArrayV2_1/element_shape:output:0-lstm_23/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Щ
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_23_while_body_21002469*'
condR
lstm_23_while_cond_21002468*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ѕ
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   №
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsp
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maskm
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤c
lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?Ћ
dropout_13/dropout/MulMul lstm_23/strided_slice_3:output:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:         ┤h
dropout_13/dropout/ShapeShape lstm_23/strided_slice_3:output:0*
T0*
_output_shapes
:Б
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:         ┤*
dtype0f
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>╚
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ┤_
dropout_13/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    └
dropout_13/dropout/SelectV2SelectV2#dropout_13/dropout/GreaterEqual:z:0dropout_13/dropout/Mul:z:0#dropout_13/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ┤Є
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	┤*
dtype0Ў
dense_30/MatMulMatMul$dropout_13/dropout/SelectV2:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_30/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2Z
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2X
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp*lstm_23/lstm_cell_23/MatMul/ReadVariableOp2\
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2
lstm_23/whilelstm_23/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▀
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003207

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ┤\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ┤"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ┤:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
Ы
Ч
0__inference_sequential_26_layer_call_fn_21002174
lstm_23_input
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
	unknown_2:	┤
	unknown_3:
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCalllstm_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_23_input
└9
н
while_body_21002817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_23_matmul_readvariableop_resource_0:	лI
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лC
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_23_matmul_readvariableop_resource:	лG
3while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лA
2while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0║
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лб
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0А
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лъ
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЏ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0Д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split{
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ┤u
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ј
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤r
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤z
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
Е
I
-__inference_dropout_13_layer_call_fn_21003197

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001880a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ┤:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
»$
з
while_body_21001637
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_23_21001661_0:	л1
while_lstm_cell_23_21001663_0:
┤л,
while_lstm_cell_23_21001665_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_23_21001661:	л/
while_lstm_cell_23_21001663:
┤л*
while_lstm_cell_23_21001665:	лѕб*while/lstm_cell_23/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Й
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_21001661_0while_lstm_cell_23_21001663_0while_lstm_cell_23_21001665_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ┤:         ┤:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001577r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_23/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Љ
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ┤Љ
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ┤y

while/NoOpNoOp+^while/lstm_cell_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_23_21001661while_lstm_cell_23_21001661_0"<
while_lstm_cell_23_21001663while_lstm_cell_23_21001663_0"<
while_lstm_cell_23_21001665while_lstm_cell_23_21001665_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2X
*while/lstm_cell_23/StatefulPartitionedCall*while/lstm_cell_23/StatefulPartitionedCall: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
Щ
щ
/__inference_lstm_cell_23_layer_call_fn_21003272

inputs
states_0
states_1
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
identity

identity_1

identity_2ѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ┤:         ┤:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001577p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┤r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ┤r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ┤:         ┤: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_1
└9
н
while_body_21002962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_23_matmul_readvariableop_resource_0:	лI
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лC
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_23_matmul_readvariableop_resource:	лG
3while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лA
2while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0║
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лб
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0А
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лъ
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЏ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0Д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split{
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ┤u
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ј
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤r
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤z
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
└9
н
while_body_21002672
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_23_matmul_readvariableop_resource_0:	лI
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лC
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_23_matmul_readvariableop_resource:	лG
3while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лA
2while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0║
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лб
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0А
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лъ
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЏ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0Д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split{
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ┤u
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ј
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤r
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤z
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
ь
Є
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001429

inputs

states
states_11
matmul_readvariableop_resource:	л4
 matmul_1_readvariableop_resource:
┤л.
biasadd_readvariableop_resource:	л
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	л*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         лs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ┤V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ┤O
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ┤`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ┤U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ┤L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ┤d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ┤Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ┤Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ┤:         ┤: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ┤
 
_user_specified_namestates:PL
(
_output_shapes
:         ┤
 
_user_specified_namestates
ЂC
н

lstm_23_while_body_21002469,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:	лQ
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лK
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorL
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:	лO
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лI
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpб0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpб2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpљ
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Г
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0м
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л▓
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0╣
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лХ
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лФ
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0┐
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лl
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :І
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitІ
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤Ї
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Ъ
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*(
_output_shapes
:         ┤Ё
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤▒
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤д
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤Ї
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤ѓ
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤х
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤z
8lstm_23/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ї
2lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_23_while_placeholder_1Alstm_23/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_23/while/addAddV2lstm_23_while_placeholderlstm_23/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: і
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: q
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: ъ
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: њ
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:         ┤њ
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:         ┤­
lstm_23/while/NoOpNoOp2^lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1^lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp3^lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_23_while_identitylstm_23/while/Identity:output:0"=
lstm_23_while_identity_1!lstm_23/while/Identity_1:output:0"=
lstm_23_while_identity_2!lstm_23/while/Identity_2:output:0"=
lstm_23_while_identity_3!lstm_23/while/Identity_3:output:0"=
lstm_23_while_identity_4!lstm_23/while/Identity_4:output:0"=
lstm_23_while_identity_5!lstm_23/while/Identity_5:output:0"P
%lstm_23_while_lstm_23_strided_slice_1'lstm_23_while_lstm_23_strided_slice_1_0"z
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"|
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"x
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"╚
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2f
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2d
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2h
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
в]
Њ
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002409

inputsF
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:	лI
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:
┤лC
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	л:
'dense_30_matmul_readvariableop_resource:	┤6
(dense_30_biasadd_readvariableop_resource:
identityѕбdense_30/BiasAdd/ReadVariableOpбdense_30/MatMul/ReadVariableOpб+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpб*lstm_23/lstm_cell_23/MatMul/ReadVariableOpб,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpбlstm_23/whileC
lstm_23/ShapeShapeinputs*
T0*
_output_shapes
:e
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤І
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ё
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*(
_output_shapes
:         ┤[
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤Ј
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    І
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ┤k
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          }
lstm_23/transpose	Transposeinputslstm_23/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:g
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЪ
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0«
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лц
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0е
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лц
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЮ
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Г
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лf
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :щ
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤Ђ
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤љ
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*(
_output_shapes
:         ┤y
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ъ
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ћ
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤Ђ
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤v
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Б
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤v
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   f
$lstm_23/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_23/TensorArrayV2_1TensorListReserve.lstm_23/TensorArrayV2_1/element_shape:output:0-lstm_23/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Щ
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_23_while_body_21002317*'
condR
lstm_23_while_cond_21002316*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ѕ
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   №
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsp
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maskm
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤c
lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
dropout_13/IdentityIdentity lstm_23/strided_slice_3:output:0*
T0*(
_output_shapes
:         ┤Є
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	┤*
dtype0Љ
dense_30/MatMulMatMuldropout_13/Identity:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_30/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2Z
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2X
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp*lstm_23/lstm_cell_23/MatMul/ReadVariableOp2\
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2
lstm_23/whilelstm_23/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ш
Ѕ
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003336

inputs
states_0
states_11
matmul_readvariableop_resource:	л4
 matmul_1_readvariableop_resource:
┤л.
biasadd_readvariableop_resource:	л
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	л*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         лs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ┤V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ┤O
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ┤`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ┤U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ┤L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ┤d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ┤Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ┤Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ┤:         ┤: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_1
ч
f
-__inference_dropout_13_layer_call_fn_21003202

inputs
identityѕбStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001942p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ┤22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
¤5
к

!__inference__traced_save_21003428
file_prefix.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop:
6savev2_lstm_23_lstm_cell_23_kernel_read_readvariableopD
@savev2_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop8
4savev2_lstm_23_lstm_cell_23_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_lstm_23_lstm_cell_23_kernel_read_readvariableopA
=savev2_adam_v_lstm_23_lstm_cell_23_kernel_read_readvariableopK
Gsavev2_adam_m_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableopK
Gsavev2_adam_v_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop?
;savev2_adam_m_lstm_23_lstm_cell_23_bias_read_readvariableop?
;savev2_adam_v_lstm_23_lstm_cell_23_bias_read_readvariableop5
1savev2_adam_m_dense_30_kernel_read_readvariableop5
1savev2_adam_v_dense_30_kernel_read_readvariableop3
/savev2_adam_m_dense_30_bias_read_readvariableop3
/savev2_adam_v_dense_30_bias_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: џ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*├	
value╣	BХ	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЮ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B в

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop6savev2_lstm_23_lstm_cell_23_kernel_read_readvariableop@savev2_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop4savev2_lstm_23_lstm_cell_23_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_lstm_23_lstm_cell_23_kernel_read_readvariableop=savev2_adam_v_lstm_23_lstm_cell_23_kernel_read_readvariableopGsavev2_adam_m_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableopGsavev2_adam_v_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop;savev2_adam_m_lstm_23_lstm_cell_23_bias_read_readvariableop;savev2_adam_v_lstm_23_lstm_cell_23_bias_read_readvariableop1savev2_adam_m_dense_30_kernel_read_readvariableop1savev2_adam_v_dense_30_kernel_read_readvariableop/savev2_adam_m_dense_30_bias_read_readvariableop/savev2_adam_v_dense_30_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *&
dtypes
2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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

identity_1Identity_1:output:0*Х
_input_shapesц
А: :	┤::	л:
┤л:л: : :	л:	л:
┤л:
┤л:л:л:	┤:	┤::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	┤: 

_output_shapes
::%!

_output_shapes
:	л:&"
 
_output_shapes
:
┤л:!

_output_shapes	
:л:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	л:%	!

_output_shapes
:	л:&
"
 
_output_shapes
:
┤л:&"
 
_output_shapes
:
┤л:!

_output_shapes	
:л:!

_output_shapes	
:л:%!

_output_shapes
:	┤:%!

_output_shapes
:	┤: 
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
: :

_output_shapes
: :

_output_shapes
: 
Д9
ј
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001514

inputs(
lstm_cell_23_21001430:	л)
lstm_cell_23_21001432:
┤л$
lstm_cell_23_21001434:	л
identityѕб$lstm_cell_23/StatefulPartitionedCallбwhile;
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskђ
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_21001430lstm_cell_23_21001432lstm_cell_23_21001434*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ┤:         ┤:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001429n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_21001430lstm_cell_23_21001432lstm_cell_23_21001434*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21001444*
condR
while_cond_21001443*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤u
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
├
═
while_cond_21002671
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21002671___redundant_placeholder06
2while_while_cond_21002671___redundant_placeholder16
2while_while_cond_21002671___redundant_placeholder26
2while_while_cond_21002671___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
└9
н
while_body_21002018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_23_matmul_readvariableop_resource_0:	лI
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лC
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_23_matmul_readvariableop_resource:	лG
3while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лA
2while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0║
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лб
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0А
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лъ
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЏ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0Д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split{
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ┤u
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ј
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤r
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤z
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
ч
Є
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002146

inputs#
lstm_23_21002132:	л$
lstm_23_21002134:
┤л
lstm_23_21002136:	л$
dense_30_21002140:	┤
dense_30_21002142:
identityѕб dense_30/StatefulPartitionedCallб"dropout_13/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallЄ
lstm_23/StatefulPartitionedCallStatefulPartitionedCallinputslstm_23_21002132lstm_23_21002134lstm_23_21002136*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002103ы
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001942Џ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_30_21002140dense_30_21002142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_21001892x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ░
NoOpNoOp!^dense_30/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
■
И
*__inference_lstm_23_layer_call_fn_21002612

inputs
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002103p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
├
═
while_cond_21001443
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21001443___redundant_placeholder06
2while_while_cond_21001443___redundant_placeholder16
2while_while_cond_21001443___redundant_placeholder26
2while_while_cond_21001443___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
С
ж
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002191
lstm_23_input#
lstm_23_21002177:	л$
lstm_23_21002179:
┤л
lstm_23_21002181:	л$
dense_30_21002185:	┤
dense_30_21002187:
identityѕб dense_30/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallј
lstm_23/StatefulPartitionedCallStatefulPartitionedCalllstm_23_inputlstm_23_21002177lstm_23_21002179lstm_23_21002181*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001867р
dropout_13/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001880Њ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_30_21002185dense_30_21002187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_21001892x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         І
NoOpNoOp!^dense_30/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_23_input
»$
з
while_body_21001444
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_23_21001468_0:	л1
while_lstm_cell_23_21001470_0:
┤л,
while_lstm_cell_23_21001472_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_23_21001468:	л/
while_lstm_cell_23_21001470:
┤л*
while_lstm_cell_23_21001472:	лѕб*while/lstm_cell_23/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Й
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_21001468_0while_lstm_cell_23_21001470_0while_lstm_cell_23_21001472_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ┤:         ┤:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001429r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_23/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Љ
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ┤Љ
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ┤y

while/NoOpNoOp+^while/lstm_cell_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_23_21001468while_lstm_cell_23_21001468_0"<
while_lstm_cell_23_21001470while_lstm_cell_23_21001470_0"<
while_lstm_cell_23_21001472while_lstm_cell_23_21001472_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2X
*while/lstm_cell_23/StatefulPartitionedCall*while/lstm_cell_23/StatefulPartitionedCall: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
├
═
while_cond_21002961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21002961___redundant_placeholder06
2while_while_cond_21002961___redundant_placeholder16
2while_while_cond_21002961___redundant_placeholder26
2while_while_cond_21002961___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
Ћ

g
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001942

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ┤C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ┤*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ┤T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ћ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         ┤b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         ┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ┤:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
Ы
Ч
0__inference_sequential_26_layer_call_fn_21001912
lstm_23_input
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
	unknown_2:	┤
	unknown_3:
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCalllstm_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_sequential_26_layer_call_and_return_conditional_losses_21001899o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_23_input
├
═
while_cond_21003106
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21003106___redundant_placeholder06
2while_while_cond_21003106___redundant_placeholder16
2while_while_cond_21003106___redundant_placeholder26
2while_while_cond_21003106___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
╩K
ъ
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003047

inputs>
+lstm_cell_23_matmul_readvariableop_resource:	лA
-lstm_cell_23_matmul_1_readvariableop_resource:
┤л;
,lstm_cell_23_biasadd_readvariableop_resource:	л
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0ќ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лћ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0љ
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лї
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЇ
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splito
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤x
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ┤i
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤|
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤f
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤І
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : і
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21002962*
condR
while_cond_21002961*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤└
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
└
Ы
&__inference_signature_wrapper_21002227
lstm_23_input
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
	unknown_2:	┤
	unknown_3:
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCalllstm_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference__wrapped_model_21001362o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_23_input
▀
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001880

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ┤\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ┤"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ┤:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
г
Ё
)sequential_26_lstm_23_while_cond_21001269H
Dsequential_26_lstm_23_while_sequential_26_lstm_23_while_loop_counterN
Jsequential_26_lstm_23_while_sequential_26_lstm_23_while_maximum_iterations+
'sequential_26_lstm_23_while_placeholder-
)sequential_26_lstm_23_while_placeholder_1-
)sequential_26_lstm_23_while_placeholder_2-
)sequential_26_lstm_23_while_placeholder_3J
Fsequential_26_lstm_23_while_less_sequential_26_lstm_23_strided_slice_1b
^sequential_26_lstm_23_while_sequential_26_lstm_23_while_cond_21001269___redundant_placeholder0b
^sequential_26_lstm_23_while_sequential_26_lstm_23_while_cond_21001269___redundant_placeholder1b
^sequential_26_lstm_23_while_sequential_26_lstm_23_while_cond_21001269___redundant_placeholder2b
^sequential_26_lstm_23_while_sequential_26_lstm_23_while_cond_21001269___redundant_placeholder3(
$sequential_26_lstm_23_while_identity
║
 sequential_26/lstm_23/while/LessLess'sequential_26_lstm_23_while_placeholderFsequential_26_lstm_23_while_less_sequential_26_lstm_23_strided_slice_1*
T0*
_output_shapes
: w
$sequential_26/lstm_23/while/IdentityIdentity$sequential_26/lstm_23/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_26_lstm_23_while_identity-sequential_26/lstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
лS
ќ
)sequential_26_lstm_23_while_body_21001270H
Dsequential_26_lstm_23_while_sequential_26_lstm_23_while_loop_counterN
Jsequential_26_lstm_23_while_sequential_26_lstm_23_while_maximum_iterations+
'sequential_26_lstm_23_while_placeholder-
)sequential_26_lstm_23_while_placeholder_1-
)sequential_26_lstm_23_while_placeholder_2-
)sequential_26_lstm_23_while_placeholder_3G
Csequential_26_lstm_23_while_sequential_26_lstm_23_strided_slice_1_0Ѓ
sequential_26_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_26_lstm_23_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_26_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:	л_
Ksequential_26_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лY
Jsequential_26_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	л(
$sequential_26_lstm_23_while_identity*
&sequential_26_lstm_23_while_identity_1*
&sequential_26_lstm_23_while_identity_2*
&sequential_26_lstm_23_while_identity_3*
&sequential_26_lstm_23_while_identity_4*
&sequential_26_lstm_23_while_identity_5E
Asequential_26_lstm_23_while_sequential_26_lstm_23_strided_slice_1Ђ
}sequential_26_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_26_lstm_23_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_26_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:	л]
Isequential_26_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лW
Hsequential_26_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб?sequential_26/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpб>sequential_26/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpб@sequential_26/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpъ
Msequential_26/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?sequential_26/lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_26_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_26_lstm_23_tensorarrayunstack_tensorlistfromtensor_0'sequential_26_lstm_23_while_placeholderVsequential_26/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0╔
>sequential_26/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOpIsequential_26_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0Ч
/sequential_26/lstm_23/while/lstm_cell_23/MatMulMatMulFsequential_26/lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_26/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л╬
@sequential_26/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpKsequential_26_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0с
1sequential_26/lstm_23/while/lstm_cell_23/MatMul_1MatMul)sequential_26_lstm_23_while_placeholder_2Hsequential_26/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лЯ
,sequential_26/lstm_23/while/lstm_cell_23/addAddV29sequential_26/lstm_23/while/lstm_cell_23/MatMul:product:0;sequential_26/lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лК
?sequential_26/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpJsequential_26_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0ж
0sequential_26/lstm_23/while/lstm_cell_23/BiasAddBiasAdd0sequential_26/lstm_23/while/lstm_cell_23/add:z:0Gsequential_26/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лz
8sequential_26/lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :х
.sequential_26/lstm_23/while/lstm_cell_23/splitSplitAsequential_26/lstm_23/while/lstm_cell_23/split/split_dim:output:09sequential_26/lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitД
0sequential_26/lstm_23/while/lstm_cell_23/SigmoidSigmoid7sequential_26/lstm_23/while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤Е
2sequential_26/lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid7sequential_26/lstm_23/while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤╔
,sequential_26/lstm_23/while/lstm_cell_23/mulMul6sequential_26/lstm_23/while/lstm_cell_23/Sigmoid_1:y:0)sequential_26_lstm_23_while_placeholder_3*
T0*(
_output_shapes
:         ┤А
-sequential_26/lstm_23/while/lstm_cell_23/ReluRelu7sequential_26/lstm_23/while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤█
.sequential_26/lstm_23/while/lstm_cell_23/mul_1Mul4sequential_26/lstm_23/while/lstm_cell_23/Sigmoid:y:0;sequential_26/lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤л
.sequential_26/lstm_23/while/lstm_cell_23/add_1AddV20sequential_26/lstm_23/while/lstm_cell_23/mul:z:02sequential_26/lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤Е
2sequential_26/lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid7sequential_26/lstm_23/while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤ъ
/sequential_26/lstm_23/while/lstm_cell_23/Relu_1Relu2sequential_26/lstm_23/while/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤▀
.sequential_26/lstm_23/while/lstm_cell_23/mul_2Mul6sequential_26/lstm_23/while/lstm_cell_23/Sigmoid_2:y:0=sequential_26/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤ѕ
Fsequential_26/lstm_23/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ┼
@sequential_26/lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_26_lstm_23_while_placeholder_1Osequential_26/lstm_23/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_26/lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмc
!sequential_26/lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ъ
sequential_26/lstm_23/while/addAddV2'sequential_26_lstm_23_while_placeholder*sequential_26/lstm_23/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_26/lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_26/lstm_23/while/add_1AddV2Dsequential_26_lstm_23_while_sequential_26_lstm_23_while_loop_counter,sequential_26/lstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: Џ
$sequential_26/lstm_23/while/IdentityIdentity%sequential_26/lstm_23/while/add_1:z:0!^sequential_26/lstm_23/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_26/lstm_23/while/Identity_1IdentityJsequential_26_lstm_23_while_sequential_26_lstm_23_while_maximum_iterations!^sequential_26/lstm_23/while/NoOp*
T0*
_output_shapes
: Џ
&sequential_26/lstm_23/while/Identity_2Identity#sequential_26/lstm_23/while/add:z:0!^sequential_26/lstm_23/while/NoOp*
T0*
_output_shapes
: ╚
&sequential_26/lstm_23/while/Identity_3IdentityPsequential_26/lstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_26/lstm_23/while/NoOp*
T0*
_output_shapes
: ╝
&sequential_26/lstm_23/while/Identity_4Identity2sequential_26/lstm_23/while/lstm_cell_23/mul_2:z:0!^sequential_26/lstm_23/while/NoOp*
T0*(
_output_shapes
:         ┤╝
&sequential_26/lstm_23/while/Identity_5Identity2sequential_26/lstm_23/while/lstm_cell_23/add_1:z:0!^sequential_26/lstm_23/while/NoOp*
T0*(
_output_shapes
:         ┤е
 sequential_26/lstm_23/while/NoOpNoOp@^sequential_26/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp?^sequential_26/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpA^sequential_26/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_26_lstm_23_while_identity-sequential_26/lstm_23/while/Identity:output:0"Y
&sequential_26_lstm_23_while_identity_1/sequential_26/lstm_23/while/Identity_1:output:0"Y
&sequential_26_lstm_23_while_identity_2/sequential_26/lstm_23/while/Identity_2:output:0"Y
&sequential_26_lstm_23_while_identity_3/sequential_26/lstm_23/while/Identity_3:output:0"Y
&sequential_26_lstm_23_while_identity_4/sequential_26/lstm_23/while/Identity_4:output:0"Y
&sequential_26_lstm_23_while_identity_5/sequential_26/lstm_23/while/Identity_5:output:0"ќ
Hsequential_26_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resourceJsequential_26_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"ў
Isequential_26_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resourceKsequential_26_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"ћ
Gsequential_26_lstm_23_while_lstm_cell_23_matmul_readvariableop_resourceIsequential_26_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"ѕ
Asequential_26_lstm_23_while_sequential_26_lstm_23_strided_slice_1Csequential_26_lstm_23_while_sequential_26_lstm_23_strided_slice_1_0"ђ
}sequential_26_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_26_lstm_23_tensorarrayunstack_tensorlistfromtensorsequential_26_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_26_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2ѓ
?sequential_26/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp?sequential_26/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2ђ
>sequential_26/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp>sequential_26/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2ё
@sequential_26/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp@sequential_26/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
¤
Р
K__inference_sequential_26_layer_call_and_return_conditional_losses_21001899

inputs#
lstm_23_21001868:	л$
lstm_23_21001870:
┤л
lstm_23_21001872:	л$
dense_30_21001893:	┤
dense_30_21001895:
identityѕб dense_30/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallЄ
lstm_23/StatefulPartitionedCallStatefulPartitionedCallinputslstm_23_21001868lstm_23_21001870lstm_23_21001872*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001867р
dropout_13/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001880Њ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_30_21001893dense_30_21001895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_21001892x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         І
NoOpNoOp!^dense_30/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
└9
н
while_body_21003107
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_23_matmul_readvariableop_resource_0:	лI
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лC
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_23_matmul_readvariableop_resource:	лG
3while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лA
2while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб)while/lstm_cell_23/BiasAdd/ReadVariableOpб(while/lstm_cell_23/MatMul/ReadVariableOpб*while/lstm_cell_23/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Ю
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0║
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лб
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0А
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лъ
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЏ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0Д
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_split{
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Є
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ┤u
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Ў
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤ј
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤}
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤r
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤Ю
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: z
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤z
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ┤л

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
├
═
while_cond_21001781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21001781___redundant_placeholder06
2while_while_cond_21001781___redundant_placeholder16
2while_while_cond_21001781___redundant_placeholder26
2while_while_cond_21001781___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
љ
ј
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002208
lstm_23_input#
lstm_23_21002194:	л$
lstm_23_21002196:
┤л
lstm_23_21002198:	л$
dense_30_21002202:	┤
dense_30_21002204:
identityѕб dense_30/StatefulPartitionedCallб"dropout_13/StatefulPartitionedCallбlstm_23/StatefulPartitionedCallј
lstm_23/StatefulPartitionedCallStatefulPartitionedCalllstm_23_inputlstm_23_21002194lstm_23_21002196lstm_23_21002198*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002103ы
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ┤* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_21001942Џ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_30_21002202dense_30_21002204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_21001892x
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ░
NoOpNoOp!^dense_30/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_23_input
Щ
щ
/__inference_lstm_cell_23_layer_call_fn_21003255

inputs
states_0
states_1
unknown:	л
	unknown_0:
┤л
	unknown_1:	л
identity

identity_1

identity_2ѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ┤:         ┤:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001429p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ┤r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ┤r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ┤`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ┤:         ┤: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_1
Д9
ј
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001707

inputs(
lstm_cell_23_21001623:	л)
lstm_cell_23_21001625:
┤л$
lstm_cell_23_21001627:	л
identityѕб$lstm_cell_23/StatefulPartitionedCallбwhile;
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskђ
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_21001623lstm_cell_23_21001625lstm_cell_23_21001627*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ┤:         ┤:         ┤*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001577n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_21001623lstm_cell_23_21001625lstm_cell_23_21001627*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21001637*
condR
while_cond_21001636*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤u
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ш
Ѕ
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003304

inputs
states_0
states_11
matmul_readvariableop_resource:	л4
 matmul_1_readvariableop_resource:
┤л.
biasadd_readvariableop_resource:	л
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	л*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         лs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ┤V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ┤O
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ┤`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ┤U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ┤L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ┤d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ┤Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ┤Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ┤:         ┤: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_0:RN
(
_output_shapes
:         ┤
"
_user_specified_name
states_1
ц

ь
lstm_23_while_cond_21002316,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1F
Blstm_23_while_lstm_23_while_cond_21002316___redundant_placeholder0F
Blstm_23_while_lstm_23_while_cond_21002316___redundant_placeholder1F
Blstm_23_while_lstm_23_while_cond_21002316___redundant_placeholder2F
Blstm_23_while_lstm_23_while_cond_21002316___redundant_placeholder3
lstm_23_while_identity
ѓ
lstm_23/while/LessLesslstm_23_while_placeholder*lstm_23_while_less_lstm_23_strided_slice_1*
T0*
_output_shapes
: [
lstm_23/while/IdentityIdentitylstm_23/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_23_while_identitylstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
»q
ї
#__inference__wrapped_model_21001362
lstm_23_inputT
Asequential_26_lstm_23_lstm_cell_23_matmul_readvariableop_resource:	лW
Csequential_26_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:
┤лQ
Bsequential_26_lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	лH
5sequential_26_dense_30_matmul_readvariableop_resource:	┤D
6sequential_26_dense_30_biasadd_readvariableop_resource:
identityѕб-sequential_26/dense_30/BiasAdd/ReadVariableOpб,sequential_26/dense_30/MatMul/ReadVariableOpб9sequential_26/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpб8sequential_26/lstm_23/lstm_cell_23/MatMul/ReadVariableOpб:sequential_26/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpбsequential_26/lstm_23/whileX
sequential_26/lstm_23/ShapeShapelstm_23_input*
T0*
_output_shapes
:s
)sequential_26/lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_26/lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_26/lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_26/lstm_23/strided_sliceStridedSlice$sequential_26/lstm_23/Shape:output:02sequential_26/lstm_23/strided_slice/stack:output:04sequential_26/lstm_23/strided_slice/stack_1:output:04sequential_26/lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
$sequential_26/lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤х
"sequential_26/lstm_23/zeros/packedPack,sequential_26/lstm_23/strided_slice:output:0-sequential_26/lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_26/lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    »
sequential_26/lstm_23/zerosFill+sequential_26/lstm_23/zeros/packed:output:0*sequential_26/lstm_23/zeros/Const:output:0*
T0*(
_output_shapes
:         ┤i
&sequential_26/lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤╣
$sequential_26/lstm_23/zeros_1/packedPack,sequential_26/lstm_23/strided_slice:output:0/sequential_26/lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_26/lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    х
sequential_26/lstm_23/zeros_1Fill-sequential_26/lstm_23/zeros_1/packed:output:0,sequential_26/lstm_23/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ┤y
$sequential_26/lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
sequential_26/lstm_23/transpose	Transposelstm_23_input-sequential_26/lstm_23/transpose/perm:output:0*
T0*+
_output_shapes
:         p
sequential_26/lstm_23/Shape_1Shape#sequential_26/lstm_23/transpose:y:0*
T0*
_output_shapes
:u
+sequential_26/lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_26/lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_26/lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_26/lstm_23/strided_slice_1StridedSlice&sequential_26/lstm_23/Shape_1:output:04sequential_26/lstm_23/strided_slice_1/stack:output:06sequential_26/lstm_23/strided_slice_1/stack_1:output:06sequential_26/lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_26/lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ш
#sequential_26/lstm_23/TensorArrayV2TensorListReserve:sequential_26/lstm_23/TensorArrayV2/element_shape:output:0.sequential_26/lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмю
Ksequential_26/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       б
=sequential_26/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_26/lstm_23/transpose:y:0Tsequential_26/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмu
+sequential_26/lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_26/lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_26/lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
%sequential_26/lstm_23/strided_slice_2StridedSlice#sequential_26/lstm_23/transpose:y:04sequential_26/lstm_23/strided_slice_2/stack:output:06sequential_26/lstm_23/strided_slice_2/stack_1:output:06sequential_26/lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask╗
8sequential_26/lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOpAsequential_26_lstm_23_lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0п
)sequential_26/lstm_23/lstm_cell_23/MatMulMatMul.sequential_26/lstm_23/strided_slice_2:output:0@sequential_26/lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л└
:sequential_26/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpCsequential_26_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0м
+sequential_26/lstm_23/lstm_cell_23/MatMul_1MatMul$sequential_26/lstm_23/zeros:output:0Bsequential_26/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л╬
&sequential_26/lstm_23/lstm_cell_23/addAddV23sequential_26/lstm_23/lstm_cell_23/MatMul:product:05sequential_26/lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         л╣
9sequential_26/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpBsequential_26_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0О
*sequential_26/lstm_23/lstm_cell_23/BiasAddBiasAdd*sequential_26/lstm_23/lstm_cell_23/add:z:0Asequential_26/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лt
2sequential_26/lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Б
(sequential_26/lstm_23/lstm_cell_23/splitSplit;sequential_26/lstm_23/lstm_cell_23/split/split_dim:output:03sequential_26/lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitЏ
*sequential_26/lstm_23/lstm_cell_23/SigmoidSigmoid1sequential_26/lstm_23/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤Ю
,sequential_26/lstm_23/lstm_cell_23/Sigmoid_1Sigmoid1sequential_26/lstm_23/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤║
&sequential_26/lstm_23/lstm_cell_23/mulMul0sequential_26/lstm_23/lstm_cell_23/Sigmoid_1:y:0&sequential_26/lstm_23/zeros_1:output:0*
T0*(
_output_shapes
:         ┤Ћ
'sequential_26/lstm_23/lstm_cell_23/ReluRelu1sequential_26/lstm_23/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤╔
(sequential_26/lstm_23/lstm_cell_23/mul_1Mul.sequential_26/lstm_23/lstm_cell_23/Sigmoid:y:05sequential_26/lstm_23/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤Й
(sequential_26/lstm_23/lstm_cell_23/add_1AddV2*sequential_26/lstm_23/lstm_cell_23/mul:z:0,sequential_26/lstm_23/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤Ю
,sequential_26/lstm_23/lstm_cell_23/Sigmoid_2Sigmoid1sequential_26/lstm_23/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤њ
)sequential_26/lstm_23/lstm_cell_23/Relu_1Relu,sequential_26/lstm_23/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤═
(sequential_26/lstm_23/lstm_cell_23/mul_2Mul0sequential_26/lstm_23/lstm_cell_23/Sigmoid_2:y:07sequential_26/lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤ё
3sequential_26/lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   t
2sequential_26/lstm_23/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Є
%sequential_26/lstm_23/TensorArrayV2_1TensorListReserve<sequential_26/lstm_23/TensorArrayV2_1/element_shape:output:0;sequential_26/lstm_23/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм\
sequential_26/lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_26/lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_26/lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
sequential_26/lstm_23/whileWhile1sequential_26/lstm_23/while/loop_counter:output:07sequential_26/lstm_23/while/maximum_iterations:output:0#sequential_26/lstm_23/time:output:0.sequential_26/lstm_23/TensorArrayV2_1:handle:0$sequential_26/lstm_23/zeros:output:0&sequential_26/lstm_23/zeros_1:output:0.sequential_26/lstm_23/strided_slice_1:output:0Msequential_26/lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_26_lstm_23_lstm_cell_23_matmul_readvariableop_resourceCsequential_26_lstm_23_lstm_cell_23_matmul_1_readvariableop_resourceBsequential_26_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_26_lstm_23_while_body_21001270*5
cond-R+
)sequential_26_lstm_23_while_cond_21001269*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ќ
Fsequential_26/lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   Ў
8sequential_26/lstm_23/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_26/lstm_23/while:output:3Osequential_26/lstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elements~
+sequential_26/lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_26/lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_26/lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
%sequential_26/lstm_23/strided_slice_3StridedSliceAsequential_26/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:04sequential_26/lstm_23/strided_slice_3/stack:output:06sequential_26/lstm_23/strided_slice_3/stack_1:output:06sequential_26/lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_mask{
&sequential_26/lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┘
!sequential_26/lstm_23/transpose_1	TransposeAsequential_26/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_26/lstm_23/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤q
sequential_26/lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    љ
!sequential_26/dropout_13/IdentityIdentity.sequential_26/lstm_23/strided_slice_3:output:0*
T0*(
_output_shapes
:         ┤Б
,sequential_26/dense_30/MatMul/ReadVariableOpReadVariableOp5sequential_26_dense_30_matmul_readvariableop_resource*
_output_shapes
:	┤*
dtype0╗
sequential_26/dense_30/MatMulMatMul*sequential_26/dropout_13/Identity:output:04sequential_26/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_26/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_26_dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_26/dense_30/BiasAddBiasAdd'sequential_26/dense_30/MatMul:product:05sequential_26/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
IdentityIdentity'sequential_26/dense_30/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         э
NoOpNoOp.^sequential_26/dense_30/BiasAdd/ReadVariableOp-^sequential_26/dense_30/MatMul/ReadVariableOp:^sequential_26/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp9^sequential_26/lstm_23/lstm_cell_23/MatMul/ReadVariableOp;^sequential_26/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^sequential_26/lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         : : : : : 2^
-sequential_26/dense_30/BiasAdd/ReadVariableOp-sequential_26/dense_30/BiasAdd/ReadVariableOp2\
,sequential_26/dense_30/MatMul/ReadVariableOp,sequential_26/dense_30/MatMul/ReadVariableOp2v
9sequential_26/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp9sequential_26/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2t
8sequential_26/lstm_23/lstm_cell_23/MatMul/ReadVariableOp8sequential_26/lstm_23/lstm_cell_23/MatMul/ReadVariableOp2x
:sequential_26/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:sequential_26/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2:
sequential_26/lstm_23/whilesequential_26/lstm_23/while:Z V
+
_output_shapes
:         
'
_user_specified_namelstm_23_input
ьK
а
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002757
inputs_0>
+lstm_cell_23_matmul_readvariableop_resource:	лA
-lstm_cell_23_matmul_1_readvariableop_resource:
┤л;
,lstm_cell_23_biasadd_readvariableop_resource:	л
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile=
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0ќ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лћ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0љ
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лї
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЇ
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splito
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤x
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ┤i
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤|
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤f
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤І
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : і
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21002672*
condR
while_cond_21002671*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤└
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
ь
Є
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21001577

inputs

states
states_11
matmul_readvariableop_resource:	л4
 matmul_1_readvariableop_resource:
┤л.
biasadd_readvariableop_resource:	л
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	л*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         лs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :║
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ┤V
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ┤O
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ┤`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ┤U
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ┤W
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ┤L
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ┤d
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ┤Y
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ┤[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ┤Љ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         :         ┤:         ┤: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ┤
 
_user_specified_namestates:PL
(
_output_shapes
:         ┤
 
_user_specified_namestates
═	
Э
F__inference_dense_30_layer_call_and_return_conditional_losses_21003238

inputs1
matmul_readvariableop_resource:	┤-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	┤*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ┤: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
ЂC
н

lstm_23_while_body_21002317,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:	лQ
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:
┤лK
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	л
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorL
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:	лO
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:
┤лI
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	лѕб1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpб0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpб2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpљ
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Г
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	л*
dtype0м
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л▓
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
┤л*
dtype0╣
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лХ
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лФ
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:л*
dtype0┐
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лl
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :І
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splitІ
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤Ї
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤Ъ
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*(
_output_shapes
:         ┤Ё
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤▒
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤д
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤Ї
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤ѓ
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤х
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤z
8lstm_23/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ї
2lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_23_while_placeholder_1Alstm_23/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_23/while/addAddV2lstm_23_while_placeholderlstm_23/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: і
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: q
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: ъ
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: њ
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:         ┤њ
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:         ┤­
lstm_23/while/NoOpNoOp2^lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1^lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp3^lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_23_while_identitylstm_23/while/Identity:output:0"=
lstm_23_while_identity_1!lstm_23/while/Identity_1:output:0"=
lstm_23_while_identity_2!lstm_23/while/Identity_2:output:0"=
lstm_23_while_identity_3!lstm_23/while/Identity_3:output:0"=
lstm_23_while_identity_4!lstm_23/while/Identity_4:output:0"=
lstm_23_while_identity_5!lstm_23/while/Identity_5:output:0"P
%lstm_23_while_lstm_23_strided_slice_1'lstm_23_while_lstm_23_strided_slice_1_0"z
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"|
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"x
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"╚
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ┤:         ┤: : : : : 2f
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2d
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2h
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
: 
├
═
while_cond_21002017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_21002017___redundant_placeholder06
2while_while_cond_21002017___redundant_placeholder16
2while_while_cond_21002017___redundant_placeholder26
2while_while_cond_21002017___redundant_placeholder3
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
B: : : : :         ┤:         ┤: ::::: 
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
:         ┤:.*
(
_output_shapes
:         ┤:

_output_shapes
: :

_output_shapes
:
Ћ

g
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003219

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ┤C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ┤*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ┤T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ћ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         ┤b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         ┤"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ┤:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
╔
Ў
+__inference_dense_30_layer_call_fn_21003228

inputs
unknown:	┤
	unknown_0:
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_21001892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ┤: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ┤
 
_user_specified_nameinputs
╩K
ъ
E__inference_lstm_23_layer_call_and_return_conditional_losses_21001867

inputs>
+lstm_cell_23_matmul_readvariableop_resource:	лA
-lstm_cell_23_matmul_1_readvariableop_resource:
┤л;
,lstm_cell_23_biasadd_readvariableop_resource:	л
identityѕб#lstm_cell_23/BiasAdd/ReadVariableOpб"lstm_cell_23/MatMul/ReadVariableOpб$lstm_cell_23/MatMul_1/ReadVariableOpбwhile;
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
valueB:Л
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
B :┤s
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
:         ┤S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :┤w
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
:         ┤c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЈ
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource*
_output_shapes
:	л*
dtype0ќ
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лћ
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
┤л*
dtype0љ
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         лї
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         лЇ
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:л*
dtype0Ћ
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         л^
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :р
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ┤:         ┤:         ┤:         ┤*
	num_splito
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:         ┤x
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ┤i
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:         ┤Є
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:         ┤|
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:         ┤q
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:         ┤f
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:         ┤І
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:         ┤n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : і
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ┤:         ┤: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_21001782*
condR
while_cond_21001781*M
output_shapes<
:: : : : :         ┤:         ┤: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ┤   О
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ┤*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ┤*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ┤[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ┤└
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultД
K
lstm_23_input:
serving_default_lstm_23_input:0         <
dense_300
StatefulPartitionedCall:0         tensorflow/serving/predict:«╝
┴
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
┌
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
╝
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
╗
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
╩
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
ш
-trace_0
.trace_1
/trace_2
0trace_32і
0__inference_sequential_26_layer_call_fn_21001912
0__inference_sequential_26_layer_call_fn_21002242
0__inference_sequential_26_layer_call_fn_21002257
0__inference_sequential_26_layer_call_fn_21002174┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z-trace_0z.trace_1z/trace_2z0trace_3
р
1trace_0
2trace_1
3trace_2
4trace_32Ш
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002409
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002568
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002191
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002208┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z1trace_0z2trace_1z3trace_2z4trace_3
нBЛ
#__inference__wrapped_model_21001362lstm_23_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю
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
╣

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
Ы
Ctrace_0
Dtrace_1
Etrace_2
Ftrace_32Є
*__inference_lstm_23_layer_call_fn_21002579
*__inference_lstm_23_layer_call_fn_21002590
*__inference_lstm_23_layer_call_fn_21002601
*__inference_lstm_23_layer_call_fn_21002612н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zCtrace_0zDtrace_1zEtrace_2zFtrace_3
я
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32з
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002757
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002902
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003047
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003192н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
"
_generic_user_object
Э
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
Г
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
╦
Xtrace_0
Ytrace_12ћ
-__inference_dropout_13_layer_call_fn_21003197
-__inference_dropout_13_layer_call_fn_21003202│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zXtrace_0zYtrace_1
Ђ
Ztrace_0
[trace_12╩
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003207
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003219│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
№
atrace_02м
+__inference_dense_30_layer_call_fn_21003228б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zatrace_0
і
btrace_02ь
F__inference_dense_30_layer_call_and_return_conditional_losses_21003238б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zbtrace_0
": 	┤2dense_30/kernel
:2dense_30/bias
.:,	л2lstm_23/lstm_cell_23/kernel
9:7
┤л2%lstm_23/lstm_cell_23/recurrent_kernel
(:&л2lstm_23/lstm_cell_23/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
c0
d1
e2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѕBЁ
0__inference_sequential_26_layer_call_fn_21001912lstm_23_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
0__inference_sequential_26_layer_call_fn_21002242inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
0__inference_sequential_26_layer_call_fn_21002257inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѕBЁ
0__inference_sequential_26_layer_call_fn_21002174lstm_23_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002409inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002568inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
БBа
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002191lstm_23_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
БBа
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002208lstm_23_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
n
60
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
C
f0
h1
j2
l3
n4"
trackable_list_wrapper
C
g0
i1
k2
m3
o4"
trackable_list_wrapper
┐2╝╣
«▓ф
FullArgSpec2
args*џ'
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

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
МBл
&__inference_signature_wrapper_21002227lstm_23_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
њBЈ
*__inference_lstm_23_layer_call_fn_21002579inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
*__inference_lstm_23_layer_call_fn_21002590inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
*__inference_lstm_23_layer_call_fn_21002601inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
*__inference_lstm_23_layer_call_fn_21002612inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ГBф
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002757inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ГBф
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002902inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003047inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003192inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
┘
utrace_0
vtrace_12б
/__inference_lstm_cell_23_layer_call_fn_21003255
/__inference_lstm_cell_23_layer_call_fn_21003272й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zutrace_0zvtrace_1
Ј
wtrace_0
xtrace_12п
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003304
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003336й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zwtrace_0zxtrace_1
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
ЫB№
-__inference_dropout_13_layer_call_fn_21003197inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЫB№
-__inference_dropout_13_layer_call_fn_21003202inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЇBі
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003207inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЇBі
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003219inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▀B▄
+__inference_dense_30_layer_call_fn_21003228inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
F__inference_dense_30_layer_call_and_return_conditional_losses_21003238inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
N
y	variables
z	keras_api
	{total
	|count"
_tf_keras_metric
`
}	variables
~	keras_api
	total

ђcount
Ђ
_fn_kwargs"
_tf_keras_metric
c
ѓ	variables
Ѓ	keras_api

ёtotal

Ёcount
є
_fn_kwargs"
_tf_keras_metric
3:1	л2"Adam/m/lstm_23/lstm_cell_23/kernel
3:1	л2"Adam/v/lstm_23/lstm_cell_23/kernel
>:<
┤л2,Adam/m/lstm_23/lstm_cell_23/recurrent_kernel
>:<
┤л2,Adam/v/lstm_23/lstm_cell_23/recurrent_kernel
-:+л2 Adam/m/lstm_23/lstm_cell_23/bias
-:+л2 Adam/v/lstm_23/lstm_cell_23/bias
':%	┤2Adam/m/dense_30/kernel
':%	┤2Adam/v/dense_30/kernel
 :2Adam/m/dense_30/bias
 :2Adam/v/dense_30/bias
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
њBЈ
/__inference_lstm_cell_23_layer_call_fn_21003255inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
њBЈ
/__inference_lstm_cell_23_layer_call_fn_21003272inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ГBф
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003304inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ГBф
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003336inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
{0
|1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
/
0
ђ1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ё0
Ё1"
trackable_list_wrapper
.
ѓ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЪ
#__inference__wrapped_model_21001362x%&'#$:б7
0б-
+і(
lstm_23_input         
ф "3ф0
.
dense_30"і
dense_30         «
F__inference_dense_30_layer_call_and_return_conditional_losses_21003238d#$0б-
&б#
!і
inputs         ┤
ф ",б)
"і
tensor_0         
џ ѕ
+__inference_dense_30_layer_call_fn_21003228Y#$0б-
&б#
!і
inputs         ┤
ф "!і
unknown         ▒
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003207e4б1
*б'
!і
inputs         ┤
p 
ф "-б*
#і 
tensor_0         ┤
џ ▒
H__inference_dropout_13_layer_call_and_return_conditional_losses_21003219e4б1
*б'
!і
inputs         ┤
p
ф "-б*
#і 
tensor_0         ┤
џ І
-__inference_dropout_13_layer_call_fn_21003197Z4б1
*б'
!і
inputs         ┤
p 
ф ""і
unknown         ┤І
-__inference_dropout_13_layer_call_fn_21003202Z4б1
*б'
!і
inputs         ┤
p
ф ""і
unknown         ┤¤
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002757Ё%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "-б*
#і 
tensor_0         ┤
џ ¤
E__inference_lstm_23_layer_call_and_return_conditional_losses_21002902Ё%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "-б*
#і 
tensor_0         ┤
џ Й
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003047u%&'?б<
5б2
$і!
inputs         

 
p 

 
ф "-б*
#і 
tensor_0         ┤
џ Й
E__inference_lstm_23_layer_call_and_return_conditional_losses_21003192u%&'?б<
5б2
$і!
inputs         

 
p

 
ф "-б*
#і 
tensor_0         ┤
џ е
*__inference_lstm_23_layer_call_fn_21002579z%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ""і
unknown         ┤е
*__inference_lstm_23_layer_call_fn_21002590z%&'OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ""і
unknown         ┤ў
*__inference_lstm_23_layer_call_fn_21002601j%&'?б<
5б2
$і!
inputs         

 
p 

 
ф ""і
unknown         ┤ў
*__inference_lstm_23_layer_call_fn_21002612j%&'?б<
5б2
$і!
inputs         

 
p

 
ф ""і
unknown         ┤ж
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003304џ%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ┤
#і 
states_1         ┤
p 
ф "ЇбЅ
Ђб~
%і"

tensor_0_0         ┤
UџR
'і$
tensor_0_1_0         ┤
'і$
tensor_0_1_1         ┤
џ ж
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_21003336џ%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ┤
#і 
states_1         ┤
p
ф "ЇбЅ
Ђб~
%і"

tensor_0_0         ┤
UџR
'і$
tensor_0_1_0         ┤
'і$
tensor_0_1_1         ┤
џ ╗
/__inference_lstm_cell_23_layer_call_fn_21003255Є%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ┤
#і 
states_1         ┤
p 
ф "{бx
#і 
tensor_0         ┤
QџN
%і"

tensor_1_0         ┤
%і"

tensor_1_1         ┤╗
/__inference_lstm_cell_23_layer_call_fn_21003272Є%&'ѓб
xбu
 і
inputs         
MбJ
#і 
states_0         ┤
#і 
states_1         ┤
p
ф "{бx
#і 
tensor_0         ┤
QџN
%і"

tensor_1_0         ┤
%і"

tensor_1_1         ┤╚
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002191y%&'#$Bб?
8б5
+і(
lstm_23_input         
p 

 
ф ",б)
"і
tensor_0         
џ ╚
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002208y%&'#$Bб?
8б5
+і(
lstm_23_input         
p

 
ф ",б)
"і
tensor_0         
џ ┴
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002409r%&'#$;б8
1б.
$і!
inputs         
p 

 
ф ",б)
"і
tensor_0         
џ ┴
K__inference_sequential_26_layer_call_and_return_conditional_losses_21002568r%&'#$;б8
1б.
$і!
inputs         
p

 
ф ",б)
"і
tensor_0         
џ б
0__inference_sequential_26_layer_call_fn_21001912n%&'#$Bб?
8б5
+і(
lstm_23_input         
p 

 
ф "!і
unknown         б
0__inference_sequential_26_layer_call_fn_21002174n%&'#$Bб?
8б5
+і(
lstm_23_input         
p

 
ф "!і
unknown         Џ
0__inference_sequential_26_layer_call_fn_21002242g%&'#$;б8
1б.
$і!
inputs         
p 

 
ф "!і
unknown         Џ
0__inference_sequential_26_layer_call_fn_21002257g%&'#$;б8
1б.
$і!
inputs         
p

 
ф "!і
unknown         ┤
&__inference_signature_wrapper_21002227Ѕ%&'#$KбH
б 
Aф>
<
lstm_23_input+і(
lstm_23_input         "3ф0
.
dense_30"і
dense_30         