��1
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
�"serve*2.11.02v2.11.0-rc2-15-g6290819256d8�.
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
Adam/v/dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_87/bias
y
(Adam/v/dense_87/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_87/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_87/bias
y
(Adam/m/dense_87/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_87/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_87/kernel
�
*Adam/v/dense_87/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_87/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_87/kernel
�
*Adam/m/dense_87/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_87/kernel*
_output_shapes

:*
dtype0
�
"Adam/v/lstm_113/lstm_cell_118/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/v/lstm_113/lstm_cell_118/bias
�
6Adam/v/lstm_113/lstm_cell_118/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_113/lstm_cell_118/bias*
_output_shapes
:x*
dtype0
�
"Adam/m/lstm_113/lstm_cell_118/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/m/lstm_113/lstm_cell_118/bias
�
6Adam/m/lstm_113/lstm_cell_118/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_113/lstm_cell_118/bias*
_output_shapes
:x*
dtype0
�
.Adam/v/lstm_113/lstm_cell_118/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/v/lstm_113/lstm_cell_118/recurrent_kernel
�
BAdam/v/lstm_113/lstm_cell_118/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_113/lstm_cell_118/recurrent_kernel*
_output_shapes

:x*
dtype0
�
.Adam/m/lstm_113/lstm_cell_118/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/m/lstm_113/lstm_cell_118/recurrent_kernel
�
BAdam/m/lstm_113/lstm_cell_118/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_113/lstm_cell_118/recurrent_kernel*
_output_shapes

:x*
dtype0
�
$Adam/v/lstm_113/lstm_cell_118/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/v/lstm_113/lstm_cell_118/kernel
�
8Adam/v/lstm_113/lstm_cell_118/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_113/lstm_cell_118/kernel*
_output_shapes

:x*
dtype0
�
$Adam/m/lstm_113/lstm_cell_118/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/m/lstm_113/lstm_cell_118/kernel
�
8Adam/m/lstm_113/lstm_cell_118/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_113/lstm_cell_118/kernel*
_output_shapes

:x*
dtype0
�
"Adam/v/lstm_112/lstm_cell_117/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/v/lstm_112/lstm_cell_117/bias
�
6Adam/v/lstm_112/lstm_cell_117/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_112/lstm_cell_117/bias*
_output_shapes
:x*
dtype0
�
"Adam/m/lstm_112/lstm_cell_117/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/m/lstm_112/lstm_cell_117/bias
�
6Adam/m/lstm_112/lstm_cell_117/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_112/lstm_cell_117/bias*
_output_shapes
:x*
dtype0
�
.Adam/v/lstm_112/lstm_cell_117/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/v/lstm_112/lstm_cell_117/recurrent_kernel
�
BAdam/v/lstm_112/lstm_cell_117/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_112/lstm_cell_117/recurrent_kernel*
_output_shapes

:x*
dtype0
�
.Adam/m/lstm_112/lstm_cell_117/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/m/lstm_112/lstm_cell_117/recurrent_kernel
�
BAdam/m/lstm_112/lstm_cell_117/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_112/lstm_cell_117/recurrent_kernel*
_output_shapes

:x*
dtype0
�
$Adam/v/lstm_112/lstm_cell_117/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/v/lstm_112/lstm_cell_117/kernel
�
8Adam/v/lstm_112/lstm_cell_117/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_112/lstm_cell_117/kernel*
_output_shapes

:x*
dtype0
�
$Adam/m/lstm_112/lstm_cell_117/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/m/lstm_112/lstm_cell_117/kernel
�
8Adam/m/lstm_112/lstm_cell_117/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_112/lstm_cell_117/kernel*
_output_shapes

:x*
dtype0
�
"Adam/v/lstm_111/lstm_cell_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/v/lstm_111/lstm_cell_116/bias
�
6Adam/v/lstm_111/lstm_cell_116/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_111/lstm_cell_116/bias*
_output_shapes
:x*
dtype0
�
"Adam/m/lstm_111/lstm_cell_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/m/lstm_111/lstm_cell_116/bias
�
6Adam/m/lstm_111/lstm_cell_116/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_111/lstm_cell_116/bias*
_output_shapes
:x*
dtype0
�
.Adam/v/lstm_111/lstm_cell_116/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/v/lstm_111/lstm_cell_116/recurrent_kernel
�
BAdam/v/lstm_111/lstm_cell_116/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_111/lstm_cell_116/recurrent_kernel*
_output_shapes

:x*
dtype0
�
.Adam/m/lstm_111/lstm_cell_116/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/m/lstm_111/lstm_cell_116/recurrent_kernel
�
BAdam/m/lstm_111/lstm_cell_116/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_111/lstm_cell_116/recurrent_kernel*
_output_shapes

:x*
dtype0
�
$Adam/v/lstm_111/lstm_cell_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/v/lstm_111/lstm_cell_116/kernel
�
8Adam/v/lstm_111/lstm_cell_116/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_111/lstm_cell_116/kernel*
_output_shapes

:x*
dtype0
�
$Adam/m/lstm_111/lstm_cell_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/m/lstm_111/lstm_cell_116/kernel
�
8Adam/m/lstm_111/lstm_cell_116/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_111/lstm_cell_116/kernel*
_output_shapes

:x*
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
lstm_113/lstm_cell_118/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namelstm_113/lstm_cell_118/bias
�
/lstm_113/lstm_cell_118/bias/Read/ReadVariableOpReadVariableOplstm_113/lstm_cell_118/bias*
_output_shapes
:x*
dtype0
�
'lstm_113/lstm_cell_118/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*8
shared_name)'lstm_113/lstm_cell_118/recurrent_kernel
�
;lstm_113/lstm_cell_118/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_113/lstm_cell_118/recurrent_kernel*
_output_shapes

:x*
dtype0
�
lstm_113/lstm_cell_118/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*.
shared_namelstm_113/lstm_cell_118/kernel
�
1lstm_113/lstm_cell_118/kernel/Read/ReadVariableOpReadVariableOplstm_113/lstm_cell_118/kernel*
_output_shapes

:x*
dtype0
�
lstm_112/lstm_cell_117/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namelstm_112/lstm_cell_117/bias
�
/lstm_112/lstm_cell_117/bias/Read/ReadVariableOpReadVariableOplstm_112/lstm_cell_117/bias*
_output_shapes
:x*
dtype0
�
'lstm_112/lstm_cell_117/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*8
shared_name)'lstm_112/lstm_cell_117/recurrent_kernel
�
;lstm_112/lstm_cell_117/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_112/lstm_cell_117/recurrent_kernel*
_output_shapes

:x*
dtype0
�
lstm_112/lstm_cell_117/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*.
shared_namelstm_112/lstm_cell_117/kernel
�
1lstm_112/lstm_cell_117/kernel/Read/ReadVariableOpReadVariableOplstm_112/lstm_cell_117/kernel*
_output_shapes

:x*
dtype0
�
lstm_111/lstm_cell_116/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namelstm_111/lstm_cell_116/bias
�
/lstm_111/lstm_cell_116/bias/Read/ReadVariableOpReadVariableOplstm_111/lstm_cell_116/bias*
_output_shapes
:x*
dtype0
�
'lstm_111/lstm_cell_116/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*8
shared_name)'lstm_111/lstm_cell_116/recurrent_kernel
�
;lstm_111/lstm_cell_116/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_111/lstm_cell_116/recurrent_kernel*
_output_shapes

:x*
dtype0
�
lstm_111/lstm_cell_116/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*.
shared_namelstm_111/lstm_cell_116/kernel
�
1lstm_111/lstm_cell_116/kernel/Read/ReadVariableOpReadVariableOplstm_111/lstm_cell_116/kernel*
_output_shapes

:x*
dtype0
r
dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_87/bias
k
!dense_87/bias/Read/ReadVariableOpReadVariableOpdense_87/bias*
_output_shapes
:*
dtype0
z
dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_87/kernel
s
#dense_87/kernel/Read/ReadVariableOpReadVariableOpdense_87/kernel*
_output_shapes

:*
dtype0
�
serving_default_lstm_111_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_111_inputlstm_111/lstm_cell_116/kernel'lstm_111/lstm_cell_116/recurrent_kernellstm_111/lstm_cell_116/biaslstm_112/lstm_cell_117/kernel'lstm_112/lstm_cell_117/recurrent_kernellstm_112/lstm_cell_117/biaslstm_113/lstm_cell_118/kernel'lstm_113/lstm_cell_118/recurrent_kernellstm_113/lstm_cell_118/biasdense_87/kerneldense_87/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_23239922

NoOpNoOp
�Y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�X
value�XB�X B�X
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell
 
state_spec*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
(cell
)
state_spec*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator* 
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
R
90
:1
;2
<3
=4
>5
?6
@7
A8
79
810*
R
90
:1
;2
<3
=4
>5
?6
@7
A8
79
810*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
* 
�
O
_variables
P_iterations
Q_learning_rate
R_index_dict
S
_momentums
T_velocities
U_update_step_xla*

Vserving_default* 

90
:1
;2*

90
:1
;2*
* 
�

Wstates
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
]trace_0
^trace_1
_trace_2
`trace_3* 
6
atrace_0
btrace_1
ctrace_2
dtrace_3* 
* 
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_random_generator
l
state_size

9kernel
:recurrent_kernel
;bias*
* 

<0
=1
>2*

<0
=1
>2*
* 
�

mstates
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
strace_0
ttrace_1
utrace_2
vtrace_3* 
6
wtrace_0
xtrace_1
ytrace_2
ztrace_3* 
* 
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size

<kernel
=recurrent_kernel
>bias*
* 

?0
@1
A2*

?0
@1
A2*
* 
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size

?kernel
@recurrent_kernel
Abias*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

70
81*

70
81*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_87/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_87/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_111/lstm_cell_116/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_111/lstm_cell_116/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_111/lstm_cell_116/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_112/lstm_cell_117/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_112/lstm_cell_117/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_112/lstm_cell_117/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_113/lstm_cell_118/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_113/lstm_cell_118/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_113/lstm_cell_118/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

�0
�1*
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
�
P0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
]
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10*
]
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10*
* 
* 
* 
* 

0*
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
90
:1
;2*

90
:1
;2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 

0*
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
<0
=1
>2*

<0
=1
>2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 

(0*
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
?0
@1
A2*

?0
@1
A2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
oi
VARIABLE_VALUE$Adam/m/lstm_111/lstm_cell_116/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/lstm_111/lstm_cell_116/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/m/lstm_111/lstm_cell_116/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/v/lstm_111/lstm_cell_116/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/lstm_111/lstm_cell_116/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_111/lstm_cell_116/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/lstm_112/lstm_cell_117/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/lstm_112/lstm_cell_117/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/m/lstm_112/lstm_cell_117/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/lstm_112/lstm_cell_117/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/lstm_112/lstm_cell_117/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/lstm_112/lstm_cell_117/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/lstm_113/lstm_cell_118/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/lstm_113/lstm_cell_118/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/m/lstm_113/lstm_cell_118/recurrent_kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/lstm_113/lstm_cell_118/recurrent_kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/lstm_113/lstm_cell_118/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/lstm_113/lstm_cell_118/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_87/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_87/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_87/bias2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_87/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_87/kernel/Read/ReadVariableOp!dense_87/bias/Read/ReadVariableOp1lstm_111/lstm_cell_116/kernel/Read/ReadVariableOp;lstm_111/lstm_cell_116/recurrent_kernel/Read/ReadVariableOp/lstm_111/lstm_cell_116/bias/Read/ReadVariableOp1lstm_112/lstm_cell_117/kernel/Read/ReadVariableOp;lstm_112/lstm_cell_117/recurrent_kernel/Read/ReadVariableOp/lstm_112/lstm_cell_117/bias/Read/ReadVariableOp1lstm_113/lstm_cell_118/kernel/Read/ReadVariableOp;lstm_113/lstm_cell_118/recurrent_kernel/Read/ReadVariableOp/lstm_113/lstm_cell_118/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp8Adam/m/lstm_111/lstm_cell_116/kernel/Read/ReadVariableOp8Adam/v/lstm_111/lstm_cell_116/kernel/Read/ReadVariableOpBAdam/m/lstm_111/lstm_cell_116/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_111/lstm_cell_116/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_111/lstm_cell_116/bias/Read/ReadVariableOp6Adam/v/lstm_111/lstm_cell_116/bias/Read/ReadVariableOp8Adam/m/lstm_112/lstm_cell_117/kernel/Read/ReadVariableOp8Adam/v/lstm_112/lstm_cell_117/kernel/Read/ReadVariableOpBAdam/m/lstm_112/lstm_cell_117/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_112/lstm_cell_117/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_112/lstm_cell_117/bias/Read/ReadVariableOp6Adam/v/lstm_112/lstm_cell_117/bias/Read/ReadVariableOp8Adam/m/lstm_113/lstm_cell_118/kernel/Read/ReadVariableOp8Adam/v/lstm_113/lstm_cell_118/kernel/Read/ReadVariableOpBAdam/m/lstm_113/lstm_cell_118/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_113/lstm_cell_118/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_113/lstm_cell_118/bias/Read/ReadVariableOp6Adam/v/lstm_113/lstm_cell_118/bias/Read/ReadVariableOp*Adam/m/dense_87/kernel/Read/ReadVariableOp*Adam/v/dense_87/kernel/Read/ReadVariableOp(Adam/m/dense_87/bias/Read/ReadVariableOp(Adam/v/dense_87/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
!__inference__traced_save_23243179
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_87/kerneldense_87/biaslstm_111/lstm_cell_116/kernel'lstm_111/lstm_cell_116/recurrent_kernellstm_111/lstm_cell_116/biaslstm_112/lstm_cell_117/kernel'lstm_112/lstm_cell_117/recurrent_kernellstm_112/lstm_cell_117/biaslstm_113/lstm_cell_118/kernel'lstm_113/lstm_cell_118/recurrent_kernellstm_113/lstm_cell_118/bias	iterationlearning_rate$Adam/m/lstm_111/lstm_cell_116/kernel$Adam/v/lstm_111/lstm_cell_116/kernel.Adam/m/lstm_111/lstm_cell_116/recurrent_kernel.Adam/v/lstm_111/lstm_cell_116/recurrent_kernel"Adam/m/lstm_111/lstm_cell_116/bias"Adam/v/lstm_111/lstm_cell_116/bias$Adam/m/lstm_112/lstm_cell_117/kernel$Adam/v/lstm_112/lstm_cell_117/kernel.Adam/m/lstm_112/lstm_cell_117/recurrent_kernel.Adam/v/lstm_112/lstm_cell_117/recurrent_kernel"Adam/m/lstm_112/lstm_cell_117/bias"Adam/v/lstm_112/lstm_cell_117/bias$Adam/m/lstm_113/lstm_cell_118/kernel$Adam/v/lstm_113/lstm_cell_118/kernel.Adam/m/lstm_113/lstm_cell_118/recurrent_kernel.Adam/v/lstm_113/lstm_cell_118/recurrent_kernel"Adam/m/lstm_113/lstm_cell_118/bias"Adam/v/lstm_113/lstm_cell_118/biasAdam/m/dense_87/kernelAdam/v/dense_87/kernelAdam/m/dense_87/biasAdam/v/dense_87/biastotal_1count_1totalcount*3
Tin,
*2(*
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
$__inference__traced_restore_23243306��,
�
f
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242714

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23238828

inputs>
,lstm_cell_116_matmul_readvariableop_resource:x@
.lstm_cell_116_matmul_1_readvariableop_resource:x;
-lstm_cell_116_biasadd_readvariableop_resource:x
identity��$lstm_cell_116/BiasAdd/ReadVariableOp�#lstm_cell_116/MatMul/ReadVariableOp�%lstm_cell_116/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_116/MatMul/ReadVariableOpReadVariableOp,lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0+lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMul_1MatMulzeros:output:0-lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_116/addAddV2lstm_cell_116/MatMul:product:0 lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_116/BiasAddBiasAddlstm_cell_116/add:z:0,lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_116/SigmoidSigmoidlstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_116/mulMullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_116/ReluRelulstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_1Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_116/add_1AddV2lstm_cell_116/mul:z:0lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_116/Relu_1Relulstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_2Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_116_matmul_readvariableop_resource.lstm_cell_116_matmul_1_readvariableop_resource-lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23238744*
condR
while_cond_23238743*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_116/BiasAdd/ReadVariableOp$^lstm_cell_116/MatMul/ReadVariableOp&^lstm_cell_116/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_116/BiasAdd/ReadVariableOp$lstm_cell_116/BiasAdd/ReadVariableOp2J
#lstm_cell_116/MatMul/ReadVariableOp#lstm_cell_116/MatMul/ReadVariableOp2N
%lstm_cell_116/MatMul_1/ReadVariableOp%lstm_cell_116/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�L
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242409
inputs_0>
,lstm_cell_118_matmul_readvariableop_resource:x@
.lstm_cell_118_matmul_1_readvariableop_resource:x;
-lstm_cell_118_biasadd_readvariableop_resource:x
identity��$lstm_cell_118/BiasAdd/ReadVariableOp�#lstm_cell_118/MatMul/ReadVariableOp�%lstm_cell_118/MatMul_1/ReadVariableOp�while=
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23242324*
condR
while_cond_23242323*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
+__inference_lstm_111_layer_call_fn_23240887

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23239708s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�S
�
!__inference__traced_save_23243179
file_prefix.
*savev2_dense_87_kernel_read_readvariableop,
(savev2_dense_87_bias_read_readvariableop<
8savev2_lstm_111_lstm_cell_116_kernel_read_readvariableopF
Bsavev2_lstm_111_lstm_cell_116_recurrent_kernel_read_readvariableop:
6savev2_lstm_111_lstm_cell_116_bias_read_readvariableop<
8savev2_lstm_112_lstm_cell_117_kernel_read_readvariableopF
Bsavev2_lstm_112_lstm_cell_117_recurrent_kernel_read_readvariableop:
6savev2_lstm_112_lstm_cell_117_bias_read_readvariableop<
8savev2_lstm_113_lstm_cell_118_kernel_read_readvariableopF
Bsavev2_lstm_113_lstm_cell_118_recurrent_kernel_read_readvariableop:
6savev2_lstm_113_lstm_cell_118_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopC
?savev2_adam_m_lstm_111_lstm_cell_116_kernel_read_readvariableopC
?savev2_adam_v_lstm_111_lstm_cell_116_kernel_read_readvariableopM
Isavev2_adam_m_lstm_111_lstm_cell_116_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_111_lstm_cell_116_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_111_lstm_cell_116_bias_read_readvariableopA
=savev2_adam_v_lstm_111_lstm_cell_116_bias_read_readvariableopC
?savev2_adam_m_lstm_112_lstm_cell_117_kernel_read_readvariableopC
?savev2_adam_v_lstm_112_lstm_cell_117_kernel_read_readvariableopM
Isavev2_adam_m_lstm_112_lstm_cell_117_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_112_lstm_cell_117_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_112_lstm_cell_117_bias_read_readvariableopA
=savev2_adam_v_lstm_112_lstm_cell_117_bias_read_readvariableopC
?savev2_adam_m_lstm_113_lstm_cell_118_kernel_read_readvariableopC
?savev2_adam_v_lstm_113_lstm_cell_118_kernel_read_readvariableopM
Isavev2_adam_m_lstm_113_lstm_cell_118_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_113_lstm_cell_118_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_113_lstm_cell_118_bias_read_readvariableopA
=savev2_adam_v_lstm_113_lstm_cell_118_bias_read_readvariableop5
1savev2_adam_m_dense_87_kernel_read_readvariableop5
1savev2_adam_v_dense_87_kernel_read_readvariableop3
/savev2_adam_m_dense_87_bias_read_readvariableop3
/savev2_adam_v_dense_87_bias_read_readvariableop&
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_87_kernel_read_readvariableop(savev2_dense_87_bias_read_readvariableop8savev2_lstm_111_lstm_cell_116_kernel_read_readvariableopBsavev2_lstm_111_lstm_cell_116_recurrent_kernel_read_readvariableop6savev2_lstm_111_lstm_cell_116_bias_read_readvariableop8savev2_lstm_112_lstm_cell_117_kernel_read_readvariableopBsavev2_lstm_112_lstm_cell_117_recurrent_kernel_read_readvariableop6savev2_lstm_112_lstm_cell_117_bias_read_readvariableop8savev2_lstm_113_lstm_cell_118_kernel_read_readvariableopBsavev2_lstm_113_lstm_cell_118_recurrent_kernel_read_readvariableop6savev2_lstm_113_lstm_cell_118_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop?savev2_adam_m_lstm_111_lstm_cell_116_kernel_read_readvariableop?savev2_adam_v_lstm_111_lstm_cell_116_kernel_read_readvariableopIsavev2_adam_m_lstm_111_lstm_cell_116_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_111_lstm_cell_116_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_111_lstm_cell_116_bias_read_readvariableop=savev2_adam_v_lstm_111_lstm_cell_116_bias_read_readvariableop?savev2_adam_m_lstm_112_lstm_cell_117_kernel_read_readvariableop?savev2_adam_v_lstm_112_lstm_cell_117_kernel_read_readvariableopIsavev2_adam_m_lstm_112_lstm_cell_117_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_112_lstm_cell_117_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_112_lstm_cell_117_bias_read_readvariableop=savev2_adam_v_lstm_112_lstm_cell_117_bias_read_readvariableop?savev2_adam_m_lstm_113_lstm_cell_118_kernel_read_readvariableop?savev2_adam_v_lstm_113_lstm_cell_118_kernel_read_readvariableopIsavev2_adam_m_lstm_113_lstm_cell_118_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_113_lstm_cell_118_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_113_lstm_cell_118_bias_read_readvariableop=savev2_adam_v_lstm_113_lstm_cell_118_bias_read_readvariableop1savev2_adam_m_dense_87_kernel_read_readvariableop1savev2_adam_v_dense_87_kernel_read_readvariableop/savev2_adam_m_dense_87_bias_read_readvariableop/savev2_adam_v_dense_87_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *6
dtypes,
*2(	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :::x:x:x:x:x:x:x:x:x: : :x:x:x:x:x:x:x:x:x:x:x:x:x:x:x:x:x:x::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x:$	 

_output_shapes

:x:$
 

_output_shapes

:x: 

_output_shapes
:x:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x: 

_output_shapes
:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x: 

_output_shapes
:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x:$ 

_output_shapes

:x: 

_output_shapes
:x: 

_output_shapes
:x:$  

_output_shapes

::$! 

_output_shapes

:: "

_output_shapes
:: #

_output_shapes
::$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: 
�
�
while_cond_23238406
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23238406___redundant_placeholder06
2while_while_cond_23238406___redundant_placeholder16
2while_while_cond_23238406___redundant_placeholder26
2while_while_cond_23238406___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23241374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23241374___redundant_placeholder06
2while_while_cond_23241374___redundant_placeholder16
2while_while_cond_23241374___redundant_placeholder26
2while_while_cond_23241374___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23241231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23241231___redundant_placeholder06
2while_while_cond_23241231___redundant_placeholder16
2while_while_cond_23241231___redundant_placeholder26
2while_while_cond_23241231___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23239623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23239623___redundant_placeholder06
2while_while_cond_23239623___redundant_placeholder16
2while_while_cond_23239623___redundant_placeholder26
2while_while_cond_23239623___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�K
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239378

inputs>
,lstm_cell_118_matmul_readvariableop_resource:x@
.lstm_cell_118_matmul_1_readvariableop_resource:x;
-lstm_cell_118_biasadd_readvariableop_resource:x
identity��$lstm_cell_118/BiasAdd/ReadVariableOp�#lstm_cell_118/MatMul/ReadVariableOp�%lstm_cell_118/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23239293*
condR
while_cond_23239292*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_lstm_111_layer_call_fn_23240876

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23238828s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_23238743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23238743___redundant_placeholder06
2while_while_cond_23238743___redundant_placeholder16
2while_while_cond_23238743___redundant_placeholder26
2while_while_cond_23238743___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
ŋ
�
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240843

inputsG
5lstm_111_lstm_cell_116_matmul_readvariableop_resource:xI
7lstm_111_lstm_cell_116_matmul_1_readvariableop_resource:xD
6lstm_111_lstm_cell_116_biasadd_readvariableop_resource:xG
5lstm_112_lstm_cell_117_matmul_readvariableop_resource:xI
7lstm_112_lstm_cell_117_matmul_1_readvariableop_resource:xD
6lstm_112_lstm_cell_117_biasadd_readvariableop_resource:xG
5lstm_113_lstm_cell_118_matmul_readvariableop_resource:xI
7lstm_113_lstm_cell_118_matmul_1_readvariableop_resource:xD
6lstm_113_lstm_cell_118_biasadd_readvariableop_resource:x9
'dense_87_matmul_readvariableop_resource:6
(dense_87_biasadd_readvariableop_resource:
identity��dense_87/BiasAdd/ReadVariableOp�dense_87/MatMul/ReadVariableOp�-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp�,lstm_111/lstm_cell_116/MatMul/ReadVariableOp�.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp�lstm_111/while�-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp�,lstm_112/lstm_cell_117/MatMul/ReadVariableOp�.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp�lstm_112/while�-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp�,lstm_113/lstm_cell_118/MatMul/ReadVariableOp�.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp�lstm_113/whileD
lstm_111/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_111/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_111/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_111/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_sliceStridedSlicelstm_111/Shape:output:0%lstm_111/strided_slice/stack:output:0'lstm_111/strided_slice/stack_1:output:0'lstm_111/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_111/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/zeros/packedPacklstm_111/strided_slice:output:0 lstm_111/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_111/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_111/zerosFilllstm_111/zeros/packed:output:0lstm_111/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_111/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/zeros_1/packedPacklstm_111/strided_slice:output:0"lstm_111/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_111/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_111/zeros_1Fill lstm_111/zeros_1/packed:output:0lstm_111/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_111/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_111/transpose	Transposeinputs lstm_111/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_111/Shape_1Shapelstm_111/transpose:y:0*
T0*
_output_shapes
:h
lstm_111/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_111/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_111/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_slice_1StridedSlicelstm_111/Shape_1:output:0'lstm_111/strided_slice_1/stack:output:0)lstm_111/strided_slice_1/stack_1:output:0)lstm_111/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_111/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_111/TensorArrayV2TensorListReserve-lstm_111/TensorArrayV2/element_shape:output:0!lstm_111/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_111/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_111/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_111/transpose:y:0Glstm_111/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_111/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_111/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_111/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_slice_2StridedSlicelstm_111/transpose:y:0'lstm_111/strided_slice_2/stack:output:0)lstm_111/strided_slice_2/stack_1:output:0)lstm_111/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_111/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp5lstm_111_lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_111/lstm_cell_116/MatMulMatMul!lstm_111/strided_slice_2:output:04lstm_111/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp7lstm_111_lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_111/lstm_cell_116/MatMul_1MatMullstm_111/zeros:output:06lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_111/lstm_cell_116/addAddV2'lstm_111/lstm_cell_116/MatMul:product:0)lstm_111/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp6lstm_111_lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_111/lstm_cell_116/BiasAddBiasAddlstm_111/lstm_cell_116/add:z:05lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_111/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/lstm_cell_116/splitSplit/lstm_111/lstm_cell_116/split/split_dim:output:0'lstm_111/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_111/lstm_cell_116/SigmoidSigmoid%lstm_111/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:����������
 lstm_111/lstm_cell_116/Sigmoid_1Sigmoid%lstm_111/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/mulMul$lstm_111/lstm_cell_116/Sigmoid_1:y:0lstm_111/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_111/lstm_cell_116/ReluRelu%lstm_111/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/mul_1Mul"lstm_111/lstm_cell_116/Sigmoid:y:0)lstm_111/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/add_1AddV2lstm_111/lstm_cell_116/mul:z:0 lstm_111/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_111/lstm_cell_116/Sigmoid_2Sigmoid%lstm_111/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������y
lstm_111/lstm_cell_116/Relu_1Relu lstm_111/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/mul_2Mul$lstm_111/lstm_cell_116/Sigmoid_2:y:0+lstm_111/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_111/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_111/TensorArrayV2_1TensorListReserve/lstm_111/TensorArrayV2_1/element_shape:output:0!lstm_111/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_111/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_111/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_111/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_111/whileWhile$lstm_111/while/loop_counter:output:0*lstm_111/while/maximum_iterations:output:0lstm_111/time:output:0!lstm_111/TensorArrayV2_1:handle:0lstm_111/zeros:output:0lstm_111/zeros_1:output:0!lstm_111/strided_slice_1:output:0@lstm_111/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_111_lstm_cell_116_matmul_readvariableop_resource7lstm_111_lstm_cell_116_matmul_1_readvariableop_resource6lstm_111_lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_111_while_body_23240465*(
cond R
lstm_111_while_cond_23240464*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_111/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_111/TensorArrayV2Stack/TensorListStackTensorListStacklstm_111/while:output:3Blstm_111/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_111/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_111/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_111/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_slice_3StridedSlice4lstm_111/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_111/strided_slice_3/stack:output:0)lstm_111/strided_slice_3/stack_1:output:0)lstm_111/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_111/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_111/transpose_1	Transpose4lstm_111/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_111/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_111/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_112/ShapeShapelstm_111/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_112/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_112/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_112/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_sliceStridedSlicelstm_112/Shape:output:0%lstm_112/strided_slice/stack:output:0'lstm_112/strided_slice/stack_1:output:0'lstm_112/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_112/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/zeros/packedPacklstm_112/strided_slice:output:0 lstm_112/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_112/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_112/zerosFilllstm_112/zeros/packed:output:0lstm_112/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_112/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/zeros_1/packedPacklstm_112/strided_slice:output:0"lstm_112/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_112/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_112/zeros_1Fill lstm_112/zeros_1/packed:output:0lstm_112/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_112/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_112/transpose	Transposelstm_111/transpose_1:y:0 lstm_112/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_112/Shape_1Shapelstm_112/transpose:y:0*
T0*
_output_shapes
:h
lstm_112/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_112/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_112/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_slice_1StridedSlicelstm_112/Shape_1:output:0'lstm_112/strided_slice_1/stack:output:0)lstm_112/strided_slice_1/stack_1:output:0)lstm_112/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_112/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_112/TensorArrayV2TensorListReserve-lstm_112/TensorArrayV2/element_shape:output:0!lstm_112/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_112/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_112/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_112/transpose:y:0Glstm_112/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_112/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_112/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_112/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_slice_2StridedSlicelstm_112/transpose:y:0'lstm_112/strided_slice_2/stack:output:0)lstm_112/strided_slice_2/stack_1:output:0)lstm_112/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_112/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp5lstm_112_lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_112/lstm_cell_117/MatMulMatMul!lstm_112/strided_slice_2:output:04lstm_112/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp7lstm_112_lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_112/lstm_cell_117/MatMul_1MatMullstm_112/zeros:output:06lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_112/lstm_cell_117/addAddV2'lstm_112/lstm_cell_117/MatMul:product:0)lstm_112/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp6lstm_112_lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_112/lstm_cell_117/BiasAddBiasAddlstm_112/lstm_cell_117/add:z:05lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_112/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/lstm_cell_117/splitSplit/lstm_112/lstm_cell_117/split/split_dim:output:0'lstm_112/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_112/lstm_cell_117/SigmoidSigmoid%lstm_112/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:����������
 lstm_112/lstm_cell_117/Sigmoid_1Sigmoid%lstm_112/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/mulMul$lstm_112/lstm_cell_117/Sigmoid_1:y:0lstm_112/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_112/lstm_cell_117/ReluRelu%lstm_112/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/mul_1Mul"lstm_112/lstm_cell_117/Sigmoid:y:0)lstm_112/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/add_1AddV2lstm_112/lstm_cell_117/mul:z:0 lstm_112/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_112/lstm_cell_117/Sigmoid_2Sigmoid%lstm_112/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������y
lstm_112/lstm_cell_117/Relu_1Relu lstm_112/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/mul_2Mul$lstm_112/lstm_cell_117/Sigmoid_2:y:0+lstm_112/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_112/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_112/TensorArrayV2_1TensorListReserve/lstm_112/TensorArrayV2_1/element_shape:output:0!lstm_112/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_112/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_112/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_112/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_112/whileWhile$lstm_112/while/loop_counter:output:0*lstm_112/while/maximum_iterations:output:0lstm_112/time:output:0!lstm_112/TensorArrayV2_1:handle:0lstm_112/zeros:output:0lstm_112/zeros_1:output:0!lstm_112/strided_slice_1:output:0@lstm_112/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_112_lstm_cell_117_matmul_readvariableop_resource7lstm_112_lstm_cell_117_matmul_1_readvariableop_resource6lstm_112_lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_112_while_body_23240604*(
cond R
lstm_112_while_cond_23240603*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_112/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_112/TensorArrayV2Stack/TensorListStackTensorListStacklstm_112/while:output:3Blstm_112/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_112/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_112/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_112/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_slice_3StridedSlice4lstm_112/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_112/strided_slice_3/stack:output:0)lstm_112/strided_slice_3/stack_1:output:0)lstm_112/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_112/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_112/transpose_1	Transpose4lstm_112/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_112/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_112/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_113/ShapeShapelstm_112/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_113/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_113/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_113/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_sliceStridedSlicelstm_113/Shape:output:0%lstm_113/strided_slice/stack:output:0'lstm_113/strided_slice/stack_1:output:0'lstm_113/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_113/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/zeros/packedPacklstm_113/strided_slice:output:0 lstm_113/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_113/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_113/zerosFilllstm_113/zeros/packed:output:0lstm_113/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_113/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/zeros_1/packedPacklstm_113/strided_slice:output:0"lstm_113/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_113/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_113/zeros_1Fill lstm_113/zeros_1/packed:output:0lstm_113/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_113/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_113/transpose	Transposelstm_112/transpose_1:y:0 lstm_113/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_113/Shape_1Shapelstm_113/transpose:y:0*
T0*
_output_shapes
:h
lstm_113/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_113/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_113/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_slice_1StridedSlicelstm_113/Shape_1:output:0'lstm_113/strided_slice_1/stack:output:0)lstm_113/strided_slice_1/stack_1:output:0)lstm_113/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_113/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_113/TensorArrayV2TensorListReserve-lstm_113/TensorArrayV2/element_shape:output:0!lstm_113/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_113/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_113/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_113/transpose:y:0Glstm_113/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_113/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_113/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_113/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_slice_2StridedSlicelstm_113/transpose:y:0'lstm_113/strided_slice_2/stack:output:0)lstm_113/strided_slice_2/stack_1:output:0)lstm_113/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_113/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp5lstm_113_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_113/lstm_cell_118/MatMulMatMul!lstm_113/strided_slice_2:output:04lstm_113/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp7lstm_113_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_113/lstm_cell_118/MatMul_1MatMullstm_113/zeros:output:06lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_113/lstm_cell_118/addAddV2'lstm_113/lstm_cell_118/MatMul:product:0)lstm_113/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp6lstm_113_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_113/lstm_cell_118/BiasAddBiasAddlstm_113/lstm_cell_118/add:z:05lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_113/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/lstm_cell_118/splitSplit/lstm_113/lstm_cell_118/split/split_dim:output:0'lstm_113/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_113/lstm_cell_118/SigmoidSigmoid%lstm_113/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:����������
 lstm_113/lstm_cell_118/Sigmoid_1Sigmoid%lstm_113/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/mulMul$lstm_113/lstm_cell_118/Sigmoid_1:y:0lstm_113/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_113/lstm_cell_118/ReluRelu%lstm_113/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/mul_1Mul"lstm_113/lstm_cell_118/Sigmoid:y:0)lstm_113/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/add_1AddV2lstm_113/lstm_cell_118/mul:z:0 lstm_113/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_113/lstm_cell_118/Sigmoid_2Sigmoid%lstm_113/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������y
lstm_113/lstm_cell_118/Relu_1Relu lstm_113/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/mul_2Mul$lstm_113/lstm_cell_118/Sigmoid_2:y:0+lstm_113/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_113/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
%lstm_113/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/TensorArrayV2_1TensorListReserve/lstm_113/TensorArrayV2_1/element_shape:output:0.lstm_113/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_113/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_113/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_113/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_113/whileWhile$lstm_113/while/loop_counter:output:0*lstm_113/while/maximum_iterations:output:0lstm_113/time:output:0!lstm_113/TensorArrayV2_1:handle:0lstm_113/zeros:output:0lstm_113/zeros_1:output:0!lstm_113/strided_slice_1:output:0@lstm_113/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_113_lstm_cell_118_matmul_readvariableop_resource7lstm_113_lstm_cell_118_matmul_1_readvariableop_resource6lstm_113_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_113_while_body_23240744*(
cond R
lstm_113_while_cond_23240743*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_113/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_113/TensorArrayV2Stack/TensorListStackTensorListStacklstm_113/while:output:3Blstm_113/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsq
lstm_113/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_113/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_113/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_slice_3StridedSlice4lstm_113/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_113/strided_slice_3/stack:output:0)lstm_113/strided_slice_3/stack_1:output:0)lstm_113/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_113/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_113/transpose_1	Transpose4lstm_113/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_113/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_113/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_70/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_70/dropout/MulMul!lstm_113/strided_slice_3:output:0!dropout_70/dropout/Const:output:0*
T0*'
_output_shapes
:���������i
dropout_70/dropout/ShapeShape!lstm_113/strided_slice_3:output:0*
T0*
_output_shapes
:�
/dropout_70/dropout/random_uniform/RandomUniformRandomUniform!dropout_70/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0f
!dropout_70/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_70/dropout/GreaterEqualGreaterEqual8dropout_70/dropout/random_uniform/RandomUniform:output:0*dropout_70/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������_
dropout_70/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_70/dropout/SelectV2SelectV2#dropout_70/dropout/GreaterEqual:z:0dropout_70/dropout/Mul:z:0#dropout_70/dropout/Const_1:output:0*
T0*'
_output_shapes
:����������
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_87/MatMulMatMul$dropout_70/dropout/SelectV2:output:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_87/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp.^lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp-^lstm_111/lstm_cell_116/MatMul/ReadVariableOp/^lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp^lstm_111/while.^lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp-^lstm_112/lstm_cell_117/MatMul/ReadVariableOp/^lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp^lstm_112/while.^lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp-^lstm_113/lstm_cell_118/MatMul/ReadVariableOp/^lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp^lstm_113/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2^
-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp2\
,lstm_111/lstm_cell_116/MatMul/ReadVariableOp,lstm_111/lstm_cell_116/MatMul/ReadVariableOp2`
.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp2 
lstm_111/whilelstm_111/while2^
-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp2\
,lstm_112/lstm_cell_117/MatMul/ReadVariableOp,lstm_112/lstm_cell_117/MatMul/ReadVariableOp2`
.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp2 
lstm_112/whilelstm_112/while2^
-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp2\
,lstm_113/lstm_cell_118/MatMul/ReadVariableOp,lstm_113/lstm_cell_118/MatMul/ReadVariableOp2`
.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp2 
lstm_113/whilelstm_113/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243039

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�	
�
F__inference_dense_87_layer_call_and_return_conditional_losses_23242745

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
0__inference_sequential_89_layer_call_fn_23239976

inputs
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:x
	unknown_7:x
	unknown_8:
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241789
inputs_0>
,lstm_cell_117_matmul_readvariableop_resource:x@
.lstm_cell_117_matmul_1_readvariableop_resource:x;
-lstm_cell_117_biasadd_readvariableop_resource:x
identity��$lstm_cell_117/BiasAdd/ReadVariableOp�#lstm_cell_117/MatMul/ReadVariableOp�%lstm_cell_117/MatMul_1/ReadVariableOp�while=
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_117/MatMul/ReadVariableOpReadVariableOp,lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMulMatMulstrided_slice_2:output:0+lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMul_1MatMulzeros:output:0-lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_117/addAddV2lstm_cell_117/MatMul:product:0 lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_117/BiasAddBiasAddlstm_cell_117/add:z:0,lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_117/splitSplit&lstm_cell_117/split/split_dim:output:0lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_117/SigmoidSigmoidlstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_1Sigmoidlstm_cell_117/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_117/mulMullstm_cell_117/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_117/ReluRelulstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_1Mullstm_cell_117/Sigmoid:y:0 lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_117/add_1AddV2lstm_cell_117/mul:z:0lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_2Sigmoidlstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_117/Relu_1Relulstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_2Mullstm_cell_117/Sigmoid_2:y:0"lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_117_matmul_readvariableop_resource.lstm_cell_117_matmul_1_readvariableop_resource-lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23241705*
condR
while_cond_23241704*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp%^lstm_cell_117/BiasAdd/ReadVariableOp$^lstm_cell_117/MatMul/ReadVariableOp&^lstm_cell_117/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_117/BiasAdd/ReadVariableOp$lstm_cell_117/BiasAdd/ReadVariableOp2J
#lstm_cell_117/MatMul/ReadVariableOp#lstm_cell_117/MatMul/ReadVariableOp2N
%lstm_cell_117/MatMul_1/ReadVariableOp%lstm_cell_117/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�9
�
while_body_23242614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_118_matmul_readvariableop_resource_0:xH
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_118_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_118_matmul_readvariableop_resource:xF
4while_lstm_cell_118_matmul_1_readvariableop_resource:xA
3while_lstm_cell_118_biasadd_readvariableop_resource:x��*while/lstm_cell_118/BiasAdd/ReadVariableOp�)while/lstm_cell_118/MatMul/ReadVariableOp�+while/lstm_cell_118/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_118/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239162

inputs#
lstm_111_23238829:x#
lstm_111_23238831:x
lstm_111_23238833:x#
lstm_112_23238979:x#
lstm_112_23238981:x
lstm_112_23238983:x#
lstm_113_23239131:x#
lstm_113_23239133:x
lstm_113_23239135:x#
dense_87_23239156:
dense_87_23239158:
identity�� dense_87/StatefulPartitionedCall� lstm_111/StatefulPartitionedCall� lstm_112/StatefulPartitionedCall� lstm_113/StatefulPartitionedCall�
 lstm_111/StatefulPartitionedCallStatefulPartitionedCallinputslstm_111_23238829lstm_111_23238831lstm_111_23238833*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23238828�
 lstm_112/StatefulPartitionedCallStatefulPartitionedCall)lstm_111/StatefulPartitionedCall:output:0lstm_112_23238979lstm_112_23238981lstm_112_23238983*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238978�
 lstm_113/StatefulPartitionedCallStatefulPartitionedCall)lstm_112/StatefulPartitionedCall:output:0lstm_113_23239131lstm_113_23239133lstm_113_23239135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239130�
dropout_70/PartitionedCallPartitionedCall)lstm_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239143�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#dropout_70/PartitionedCall:output:0dense_87_23239156dense_87_23239158*
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
F__inference_dense_87_layer_call_and_return_conditional_losses_23239155x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall!^lstm_111/StatefulPartitionedCall!^lstm_112/StatefulPartitionedCall!^lstm_113/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 lstm_111/StatefulPartitionedCall lstm_111/StatefulPartitionedCall2D
 lstm_112/StatefulPartitionedCall lstm_112/StatefulPartitionedCall2D
 lstm_113/StatefulPartitionedCall lstm_113/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23239708

inputs>
,lstm_cell_116_matmul_readvariableop_resource:x@
.lstm_cell_116_matmul_1_readvariableop_resource:x;
-lstm_cell_116_biasadd_readvariableop_resource:x
identity��$lstm_cell_116/BiasAdd/ReadVariableOp�#lstm_cell_116/MatMul/ReadVariableOp�%lstm_cell_116/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_116/MatMul/ReadVariableOpReadVariableOp,lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0+lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMul_1MatMulzeros:output:0-lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_116/addAddV2lstm_cell_116/MatMul:product:0 lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_116/BiasAddBiasAddlstm_cell_116/add:z:0,lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_116/SigmoidSigmoidlstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_116/mulMullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_116/ReluRelulstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_1Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_116/add_1AddV2lstm_cell_116/mul:z:0lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_116/Relu_1Relulstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_2Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_116_matmul_readvariableop_resource.lstm_cell_116_matmul_1_readvariableop_resource-lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23239624*
condR
while_cond_23239623*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_116/BiasAdd/ReadVariableOp$^lstm_cell_116/MatMul/ReadVariableOp&^lstm_cell_116/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_116/BiasAdd/ReadVariableOp$lstm_cell_116/BiasAdd/ReadVariableOp2J
#lstm_cell_116/MatMul/ReadVariableOp#lstm_cell_116/MatMul/ReadVariableOp2N
%lstm_cell_116/MatMul_1/ReadVariableOp%lstm_cell_116/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239777

inputs#
lstm_111_23239749:x#
lstm_111_23239751:x
lstm_111_23239753:x#
lstm_112_23239756:x#
lstm_112_23239758:x
lstm_112_23239760:x#
lstm_113_23239763:x#
lstm_113_23239765:x
lstm_113_23239767:x#
dense_87_23239771:
dense_87_23239773:
identity�� dense_87/StatefulPartitionedCall�"dropout_70/StatefulPartitionedCall� lstm_111/StatefulPartitionedCall� lstm_112/StatefulPartitionedCall� lstm_113/StatefulPartitionedCall�
 lstm_111/StatefulPartitionedCallStatefulPartitionedCallinputslstm_111_23239749lstm_111_23239751lstm_111_23239753*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23239708�
 lstm_112/StatefulPartitionedCallStatefulPartitionedCall)lstm_111/StatefulPartitionedCall:output:0lstm_112_23239756lstm_112_23239758lstm_112_23239760*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23239543�
 lstm_113/StatefulPartitionedCallStatefulPartitionedCall)lstm_112/StatefulPartitionedCall:output:0lstm_113_23239763lstm_113_23239765lstm_113_23239767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239378�
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall)lstm_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239217�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall+dropout_70/StatefulPartitionedCall:output:0dense_87_23239771dense_87_23239773*
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
F__inference_dense_87_layer_call_and_return_conditional_losses_23239155x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall#^dropout_70/StatefulPartitionedCall!^lstm_111/StatefulPartitionedCall!^lstm_112/StatefulPartitionedCall!^lstm_113/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2D
 lstm_111/StatefulPartitionedCall lstm_111/StatefulPartitionedCall2D
 lstm_112/StatefulPartitionedCall lstm_112/StatefulPartitionedCall2D
 lstm_113/StatefulPartitionedCall lstm_113/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_23239459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_117_matmul_readvariableop_resource_0:xH
6while_lstm_cell_117_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_117_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_117_matmul_readvariableop_resource:xF
4while_lstm_cell_117_matmul_1_readvariableop_resource:xA
3while_lstm_cell_117_biasadd_readvariableop_resource:x��*while/lstm_cell_117/BiasAdd/ReadVariableOp�)while/lstm_cell_117/MatMul/ReadVariableOp�+while/lstm_cell_117/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_117/addAddV2$while/lstm_cell_117/MatMul:product:0&while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_117/BiasAddBiasAddwhile/lstm_cell_117/add:z:02while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_117/splitSplit,while/lstm_cell_117/split/split_dim:output:0$while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_117/SigmoidSigmoid"while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_1Sigmoid"while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mulMul!while/lstm_cell_117/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_117/ReluRelu"while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_1Mulwhile/lstm_cell_117/Sigmoid:y:0&while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/add_1AddV2while/lstm_cell_117/mul:z:0while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_2Sigmoid"while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_117/Relu_1Reluwhile/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_2Mul!while/lstm_cell_117/Sigmoid_2:y:0(while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_117/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_117/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_117/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_117/BiasAdd/ReadVariableOp*^while/lstm_cell_117/MatMul/ReadVariableOp,^while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_117_biasadd_readvariableop_resource5while_lstm_cell_117_biasadd_readvariableop_resource_0"n
4while_lstm_cell_117_matmul_1_readvariableop_resource6while_lstm_cell_117_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_117_matmul_readvariableop_resource4while_lstm_cell_117_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_117/BiasAdd/ReadVariableOp*while/lstm_cell_117/BiasAdd/ReadVariableOp2V
)while/lstm_cell_117/MatMul/ReadVariableOp)while/lstm_cell_117/MatMul/ReadVariableOp2Z
+while/lstm_cell_117/MatMul_1/ReadVariableOp+while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
I
-__inference_dropout_70_layer_call_fn_23242704

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239143`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241173
inputs_0>
,lstm_cell_116_matmul_readvariableop_resource:x@
.lstm_cell_116_matmul_1_readvariableop_resource:x;
-lstm_cell_116_biasadd_readvariableop_resource:x
identity��$lstm_cell_116/BiasAdd/ReadVariableOp�#lstm_cell_116/MatMul/ReadVariableOp�%lstm_cell_116/MatMul_1/ReadVariableOp�while=
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_116/MatMul/ReadVariableOpReadVariableOp,lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0+lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMul_1MatMulzeros:output:0-lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_116/addAddV2lstm_cell_116/MatMul:product:0 lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_116/BiasAddBiasAddlstm_cell_116/add:z:0,lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_116/SigmoidSigmoidlstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_116/mulMullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_116/ReluRelulstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_1Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_116/add_1AddV2lstm_cell_116/mul:z:0lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_116/Relu_1Relulstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_2Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_116_matmul_readvariableop_resource.lstm_cell_116_matmul_1_readvariableop_resource-lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23241089*
condR
while_cond_23241088*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp%^lstm_cell_116/BiasAdd/ReadVariableOp$^lstm_cell_116/MatMul/ReadVariableOp&^lstm_cell_116/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_116/BiasAdd/ReadVariableOp$lstm_cell_116/BiasAdd/ReadVariableOp2J
#lstm_cell_116/MatMul/ReadVariableOp#lstm_cell_116/MatMul/ReadVariableOp2N
%lstm_cell_116/MatMul_1/ReadVariableOp%lstm_cell_116/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�K
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241646
inputs_0>
,lstm_cell_117_matmul_readvariableop_resource:x@
.lstm_cell_117_matmul_1_readvariableop_resource:x;
-lstm_cell_117_biasadd_readvariableop_resource:x
identity��$lstm_cell_117/BiasAdd/ReadVariableOp�#lstm_cell_117/MatMul/ReadVariableOp�%lstm_cell_117/MatMul_1/ReadVariableOp�while=
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_117/MatMul/ReadVariableOpReadVariableOp,lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMulMatMulstrided_slice_2:output:0+lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMul_1MatMulzeros:output:0-lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_117/addAddV2lstm_cell_117/MatMul:product:0 lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_117/BiasAddBiasAddlstm_cell_117/add:z:0,lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_117/splitSplit&lstm_cell_117/split/split_dim:output:0lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_117/SigmoidSigmoidlstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_1Sigmoidlstm_cell_117/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_117/mulMullstm_cell_117/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_117/ReluRelulstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_1Mullstm_cell_117/Sigmoid:y:0 lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_117/add_1AddV2lstm_cell_117/mul:z:0lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_2Sigmoidlstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_117/Relu_1Relulstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_2Mullstm_cell_117/Sigmoid_2:y:0"lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_117_matmul_readvariableop_resource.lstm_cell_117_matmul_1_readvariableop_resource-lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23241562*
condR
while_cond_23241561*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp%^lstm_cell_117/BiasAdd/ReadVariableOp$^lstm_cell_117/MatMul/ReadVariableOp&^lstm_cell_117/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_117/BiasAdd/ReadVariableOp$lstm_cell_117/BiasAdd/ReadVariableOp2J
#lstm_cell_117/MatMul/ReadVariableOp#lstm_cell_117/MatMul/ReadVariableOp2N
%lstm_cell_117/MatMul_1/ReadVariableOp%lstm_cell_117/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
while_cond_23240945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23240945___redundant_placeholder06
2while_while_cond_23240945___redundant_placeholder16
2while_while_cond_23240945___redundant_placeholder26
2while_while_cond_23240945___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
׃
�
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240406

inputsG
5lstm_111_lstm_cell_116_matmul_readvariableop_resource:xI
7lstm_111_lstm_cell_116_matmul_1_readvariableop_resource:xD
6lstm_111_lstm_cell_116_biasadd_readvariableop_resource:xG
5lstm_112_lstm_cell_117_matmul_readvariableop_resource:xI
7lstm_112_lstm_cell_117_matmul_1_readvariableop_resource:xD
6lstm_112_lstm_cell_117_biasadd_readvariableop_resource:xG
5lstm_113_lstm_cell_118_matmul_readvariableop_resource:xI
7lstm_113_lstm_cell_118_matmul_1_readvariableop_resource:xD
6lstm_113_lstm_cell_118_biasadd_readvariableop_resource:x9
'dense_87_matmul_readvariableop_resource:6
(dense_87_biasadd_readvariableop_resource:
identity��dense_87/BiasAdd/ReadVariableOp�dense_87/MatMul/ReadVariableOp�-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp�,lstm_111/lstm_cell_116/MatMul/ReadVariableOp�.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp�lstm_111/while�-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp�,lstm_112/lstm_cell_117/MatMul/ReadVariableOp�.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp�lstm_112/while�-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp�,lstm_113/lstm_cell_118/MatMul/ReadVariableOp�.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp�lstm_113/whileD
lstm_111/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_111/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_111/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_111/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_sliceStridedSlicelstm_111/Shape:output:0%lstm_111/strided_slice/stack:output:0'lstm_111/strided_slice/stack_1:output:0'lstm_111/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_111/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/zeros/packedPacklstm_111/strided_slice:output:0 lstm_111/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_111/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_111/zerosFilllstm_111/zeros/packed:output:0lstm_111/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_111/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/zeros_1/packedPacklstm_111/strided_slice:output:0"lstm_111/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_111/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_111/zeros_1Fill lstm_111/zeros_1/packed:output:0lstm_111/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_111/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_111/transpose	Transposeinputs lstm_111/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_111/Shape_1Shapelstm_111/transpose:y:0*
T0*
_output_shapes
:h
lstm_111/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_111/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_111/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_slice_1StridedSlicelstm_111/Shape_1:output:0'lstm_111/strided_slice_1/stack:output:0)lstm_111/strided_slice_1/stack_1:output:0)lstm_111/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_111/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_111/TensorArrayV2TensorListReserve-lstm_111/TensorArrayV2/element_shape:output:0!lstm_111/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_111/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_111/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_111/transpose:y:0Glstm_111/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_111/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_111/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_111/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_slice_2StridedSlicelstm_111/transpose:y:0'lstm_111/strided_slice_2/stack:output:0)lstm_111/strided_slice_2/stack_1:output:0)lstm_111/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_111/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp5lstm_111_lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_111/lstm_cell_116/MatMulMatMul!lstm_111/strided_slice_2:output:04lstm_111/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp7lstm_111_lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_111/lstm_cell_116/MatMul_1MatMullstm_111/zeros:output:06lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_111/lstm_cell_116/addAddV2'lstm_111/lstm_cell_116/MatMul:product:0)lstm_111/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp6lstm_111_lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_111/lstm_cell_116/BiasAddBiasAddlstm_111/lstm_cell_116/add:z:05lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_111/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/lstm_cell_116/splitSplit/lstm_111/lstm_cell_116/split/split_dim:output:0'lstm_111/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_111/lstm_cell_116/SigmoidSigmoid%lstm_111/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:����������
 lstm_111/lstm_cell_116/Sigmoid_1Sigmoid%lstm_111/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/mulMul$lstm_111/lstm_cell_116/Sigmoid_1:y:0lstm_111/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_111/lstm_cell_116/ReluRelu%lstm_111/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/mul_1Mul"lstm_111/lstm_cell_116/Sigmoid:y:0)lstm_111/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/add_1AddV2lstm_111/lstm_cell_116/mul:z:0 lstm_111/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_111/lstm_cell_116/Sigmoid_2Sigmoid%lstm_111/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������y
lstm_111/lstm_cell_116/Relu_1Relu lstm_111/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_111/lstm_cell_116/mul_2Mul$lstm_111/lstm_cell_116/Sigmoid_2:y:0+lstm_111/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_111/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_111/TensorArrayV2_1TensorListReserve/lstm_111/TensorArrayV2_1/element_shape:output:0!lstm_111/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_111/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_111/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_111/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_111/whileWhile$lstm_111/while/loop_counter:output:0*lstm_111/while/maximum_iterations:output:0lstm_111/time:output:0!lstm_111/TensorArrayV2_1:handle:0lstm_111/zeros:output:0lstm_111/zeros_1:output:0!lstm_111/strided_slice_1:output:0@lstm_111/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_111_lstm_cell_116_matmul_readvariableop_resource7lstm_111_lstm_cell_116_matmul_1_readvariableop_resource6lstm_111_lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_111_while_body_23240035*(
cond R
lstm_111_while_cond_23240034*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_111/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_111/TensorArrayV2Stack/TensorListStackTensorListStacklstm_111/while:output:3Blstm_111/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_111/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_111/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_111/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_111/strided_slice_3StridedSlice4lstm_111/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_111/strided_slice_3/stack:output:0)lstm_111/strided_slice_3/stack_1:output:0)lstm_111/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_111/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_111/transpose_1	Transpose4lstm_111/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_111/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_111/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_112/ShapeShapelstm_111/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_112/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_112/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_112/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_sliceStridedSlicelstm_112/Shape:output:0%lstm_112/strided_slice/stack:output:0'lstm_112/strided_slice/stack_1:output:0'lstm_112/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_112/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/zeros/packedPacklstm_112/strided_slice:output:0 lstm_112/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_112/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_112/zerosFilllstm_112/zeros/packed:output:0lstm_112/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_112/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/zeros_1/packedPacklstm_112/strided_slice:output:0"lstm_112/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_112/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_112/zeros_1Fill lstm_112/zeros_1/packed:output:0lstm_112/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_112/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_112/transpose	Transposelstm_111/transpose_1:y:0 lstm_112/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_112/Shape_1Shapelstm_112/transpose:y:0*
T0*
_output_shapes
:h
lstm_112/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_112/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_112/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_slice_1StridedSlicelstm_112/Shape_1:output:0'lstm_112/strided_slice_1/stack:output:0)lstm_112/strided_slice_1/stack_1:output:0)lstm_112/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_112/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_112/TensorArrayV2TensorListReserve-lstm_112/TensorArrayV2/element_shape:output:0!lstm_112/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_112/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_112/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_112/transpose:y:0Glstm_112/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_112/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_112/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_112/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_slice_2StridedSlicelstm_112/transpose:y:0'lstm_112/strided_slice_2/stack:output:0)lstm_112/strided_slice_2/stack_1:output:0)lstm_112/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_112/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp5lstm_112_lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_112/lstm_cell_117/MatMulMatMul!lstm_112/strided_slice_2:output:04lstm_112/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp7lstm_112_lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_112/lstm_cell_117/MatMul_1MatMullstm_112/zeros:output:06lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_112/lstm_cell_117/addAddV2'lstm_112/lstm_cell_117/MatMul:product:0)lstm_112/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp6lstm_112_lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_112/lstm_cell_117/BiasAddBiasAddlstm_112/lstm_cell_117/add:z:05lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_112/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/lstm_cell_117/splitSplit/lstm_112/lstm_cell_117/split/split_dim:output:0'lstm_112/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_112/lstm_cell_117/SigmoidSigmoid%lstm_112/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:����������
 lstm_112/lstm_cell_117/Sigmoid_1Sigmoid%lstm_112/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/mulMul$lstm_112/lstm_cell_117/Sigmoid_1:y:0lstm_112/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_112/lstm_cell_117/ReluRelu%lstm_112/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/mul_1Mul"lstm_112/lstm_cell_117/Sigmoid:y:0)lstm_112/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/add_1AddV2lstm_112/lstm_cell_117/mul:z:0 lstm_112/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_112/lstm_cell_117/Sigmoid_2Sigmoid%lstm_112/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������y
lstm_112/lstm_cell_117/Relu_1Relu lstm_112/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_112/lstm_cell_117/mul_2Mul$lstm_112/lstm_cell_117/Sigmoid_2:y:0+lstm_112/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_112/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_112/TensorArrayV2_1TensorListReserve/lstm_112/TensorArrayV2_1/element_shape:output:0!lstm_112/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_112/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_112/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_112/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_112/whileWhile$lstm_112/while/loop_counter:output:0*lstm_112/while/maximum_iterations:output:0lstm_112/time:output:0!lstm_112/TensorArrayV2_1:handle:0lstm_112/zeros:output:0lstm_112/zeros_1:output:0!lstm_112/strided_slice_1:output:0@lstm_112/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_112_lstm_cell_117_matmul_readvariableop_resource7lstm_112_lstm_cell_117_matmul_1_readvariableop_resource6lstm_112_lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_112_while_body_23240174*(
cond R
lstm_112_while_cond_23240173*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_112/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_112/TensorArrayV2Stack/TensorListStackTensorListStacklstm_112/while:output:3Blstm_112/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_112/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_112/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_112/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_112/strided_slice_3StridedSlice4lstm_112/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_112/strided_slice_3/stack:output:0)lstm_112/strided_slice_3/stack_1:output:0)lstm_112/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_112/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_112/transpose_1	Transpose4lstm_112/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_112/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_112/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_113/ShapeShapelstm_112/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_113/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_113/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_113/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_sliceStridedSlicelstm_113/Shape:output:0%lstm_113/strided_slice/stack:output:0'lstm_113/strided_slice/stack_1:output:0'lstm_113/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_113/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/zeros/packedPacklstm_113/strided_slice:output:0 lstm_113/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_113/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_113/zerosFilllstm_113/zeros/packed:output:0lstm_113/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_113/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/zeros_1/packedPacklstm_113/strided_slice:output:0"lstm_113/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_113/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_113/zeros_1Fill lstm_113/zeros_1/packed:output:0lstm_113/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_113/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_113/transpose	Transposelstm_112/transpose_1:y:0 lstm_113/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_113/Shape_1Shapelstm_113/transpose:y:0*
T0*
_output_shapes
:h
lstm_113/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_113/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_113/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_slice_1StridedSlicelstm_113/Shape_1:output:0'lstm_113/strided_slice_1/stack:output:0)lstm_113/strided_slice_1/stack_1:output:0)lstm_113/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_113/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_113/TensorArrayV2TensorListReserve-lstm_113/TensorArrayV2/element_shape:output:0!lstm_113/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_113/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_113/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_113/transpose:y:0Glstm_113/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_113/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_113/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_113/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_slice_2StridedSlicelstm_113/transpose:y:0'lstm_113/strided_slice_2/stack:output:0)lstm_113/strided_slice_2/stack_1:output:0)lstm_113/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_113/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp5lstm_113_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_113/lstm_cell_118/MatMulMatMul!lstm_113/strided_slice_2:output:04lstm_113/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp7lstm_113_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_113/lstm_cell_118/MatMul_1MatMullstm_113/zeros:output:06lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_113/lstm_cell_118/addAddV2'lstm_113/lstm_cell_118/MatMul:product:0)lstm_113/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp6lstm_113_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_113/lstm_cell_118/BiasAddBiasAddlstm_113/lstm_cell_118/add:z:05lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_113/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/lstm_cell_118/splitSplit/lstm_113/lstm_cell_118/split/split_dim:output:0'lstm_113/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_113/lstm_cell_118/SigmoidSigmoid%lstm_113/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:����������
 lstm_113/lstm_cell_118/Sigmoid_1Sigmoid%lstm_113/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/mulMul$lstm_113/lstm_cell_118/Sigmoid_1:y:0lstm_113/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_113/lstm_cell_118/ReluRelu%lstm_113/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/mul_1Mul"lstm_113/lstm_cell_118/Sigmoid:y:0)lstm_113/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/add_1AddV2lstm_113/lstm_cell_118/mul:z:0 lstm_113/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_113/lstm_cell_118/Sigmoid_2Sigmoid%lstm_113/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������y
lstm_113/lstm_cell_118/Relu_1Relu lstm_113/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_113/lstm_cell_118/mul_2Mul$lstm_113/lstm_cell_118/Sigmoid_2:y:0+lstm_113/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_113/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
%lstm_113/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/TensorArrayV2_1TensorListReserve/lstm_113/TensorArrayV2_1/element_shape:output:0.lstm_113/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_113/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_113/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_113/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_113/whileWhile$lstm_113/while/loop_counter:output:0*lstm_113/while/maximum_iterations:output:0lstm_113/time:output:0!lstm_113/TensorArrayV2_1:handle:0lstm_113/zeros:output:0lstm_113/zeros_1:output:0!lstm_113/strided_slice_1:output:0@lstm_113/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_113_lstm_cell_118_matmul_readvariableop_resource7lstm_113_lstm_cell_118_matmul_1_readvariableop_resource6lstm_113_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_113_while_body_23240314*(
cond R
lstm_113_while_cond_23240313*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_113/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_113/TensorArrayV2Stack/TensorListStackTensorListStacklstm_113/while:output:3Blstm_113/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsq
lstm_113/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_113/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_113/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_113/strided_slice_3StridedSlice4lstm_113/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_113/strided_slice_3/stack:output:0)lstm_113/strided_slice_3/stack_1:output:0)lstm_113/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_113/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_113/transpose_1	Transpose4lstm_113/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_113/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_113/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
dropout_70/IdentityIdentity!lstm_113/strided_slice_3:output:0*
T0*'
_output_shapes
:����������
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_87/MatMulMatMuldropout_70/Identity:output:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_87/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_87/BiasAdd/ReadVariableOp^dense_87/MatMul/ReadVariableOp.^lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp-^lstm_111/lstm_cell_116/MatMul/ReadVariableOp/^lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp^lstm_111/while.^lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp-^lstm_112/lstm_cell_117/MatMul/ReadVariableOp/^lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp^lstm_112/while.^lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp-^lstm_113/lstm_cell_118/MatMul/ReadVariableOp/^lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp^lstm_113/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_87/BiasAdd/ReadVariableOpdense_87/BiasAdd/ReadVariableOp2@
dense_87/MatMul/ReadVariableOpdense_87/MatMul/ReadVariableOp2^
-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp-lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp2\
,lstm_111/lstm_cell_116/MatMul/ReadVariableOp,lstm_111/lstm_cell_116/MatMul/ReadVariableOp2`
.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp.lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp2 
lstm_111/whilelstm_111/while2^
-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp-lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp2\
,lstm_112/lstm_cell_117/MatMul/ReadVariableOp,lstm_112/lstm_cell_117/MatMul/ReadVariableOp2`
.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp.lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp2 
lstm_112/whilelstm_112/while2^
-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp-lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp2\
,lstm_113/lstm_cell_118/MatMul/ReadVariableOp,lstm_113/lstm_cell_118/MatMul/ReadVariableOp2`
.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp.lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp2 
lstm_113/whilelstm_113/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239860
lstm_111_input#
lstm_111_23239832:x#
lstm_111_23239834:x
lstm_111_23239836:x#
lstm_112_23239839:x#
lstm_112_23239841:x
lstm_112_23239843:x#
lstm_113_23239846:x#
lstm_113_23239848:x
lstm_113_23239850:x#
dense_87_23239854:
dense_87_23239856:
identity�� dense_87/StatefulPartitionedCall� lstm_111/StatefulPartitionedCall� lstm_112/StatefulPartitionedCall� lstm_113/StatefulPartitionedCall�
 lstm_111/StatefulPartitionedCallStatefulPartitionedCalllstm_111_inputlstm_111_23239832lstm_111_23239834lstm_111_23239836*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23238828�
 lstm_112/StatefulPartitionedCallStatefulPartitionedCall)lstm_111/StatefulPartitionedCall:output:0lstm_112_23239839lstm_112_23239841lstm_112_23239843*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238978�
 lstm_113/StatefulPartitionedCallStatefulPartitionedCall)lstm_112/StatefulPartitionedCall:output:0lstm_113_23239846lstm_113_23239848lstm_113_23239850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239130�
dropout_70/PartitionedCallPartitionedCall)lstm_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239143�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall#dropout_70/PartitionedCall:output:0dense_87_23239854dense_87_23239856*
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
F__inference_dense_87_layer_call_and_return_conditional_losses_23239155x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall!^lstm_111/StatefulPartitionedCall!^lstm_112/StatefulPartitionedCall!^lstm_113/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2D
 lstm_111/StatefulPartitionedCall lstm_111/StatefulPartitionedCall2D
 lstm_112/StatefulPartitionedCall lstm_112/StatefulPartitionedCall2D
 lstm_113/StatefulPartitionedCall lstm_113/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_111_input
�
�
0__inference_lstm_cell_118_layer_call_fn_23242975

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�U
�
*sequential_89_lstm_113_while_body_23237533J
Fsequential_89_lstm_113_while_sequential_89_lstm_113_while_loop_counterP
Lsequential_89_lstm_113_while_sequential_89_lstm_113_while_maximum_iterations,
(sequential_89_lstm_113_while_placeholder.
*sequential_89_lstm_113_while_placeholder_1.
*sequential_89_lstm_113_while_placeholder_2.
*sequential_89_lstm_113_while_placeholder_3I
Esequential_89_lstm_113_while_sequential_89_lstm_113_strided_slice_1_0�
�sequential_89_lstm_113_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_113_tensorarrayunstack_tensorlistfromtensor_0]
Ksequential_89_lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0:x_
Msequential_89_lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0:xZ
Lsequential_89_lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0:x)
%sequential_89_lstm_113_while_identity+
'sequential_89_lstm_113_while_identity_1+
'sequential_89_lstm_113_while_identity_2+
'sequential_89_lstm_113_while_identity_3+
'sequential_89_lstm_113_while_identity_4+
'sequential_89_lstm_113_while_identity_5G
Csequential_89_lstm_113_while_sequential_89_lstm_113_strided_slice_1�
sequential_89_lstm_113_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_113_tensorarrayunstack_tensorlistfromtensor[
Isequential_89_lstm_113_while_lstm_cell_118_matmul_readvariableop_resource:x]
Ksequential_89_lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource:xX
Jsequential_89_lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource:x��Asequential_89/lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp�@sequential_89/lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp�Bsequential_89/lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp�
Nsequential_89/lstm_113/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
@sequential_89/lstm_113/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_89_lstm_113_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_113_tensorarrayunstack_tensorlistfromtensor_0(sequential_89_lstm_113_while_placeholderWsequential_89/lstm_113/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
@sequential_89/lstm_113/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpKsequential_89_lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
1sequential_89/lstm_113/while/lstm_cell_118/MatMulMatMulGsequential_89/lstm_113/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_89/lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
Bsequential_89/lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpMsequential_89_lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
3sequential_89/lstm_113/while/lstm_cell_118/MatMul_1MatMul*sequential_89_lstm_113_while_placeholder_2Jsequential_89/lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.sequential_89/lstm_113/while/lstm_cell_118/addAddV2;sequential_89/lstm_113/while/lstm_cell_118/MatMul:product:0=sequential_89/lstm_113/while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
Asequential_89/lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpLsequential_89_lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
2sequential_89/lstm_113/while/lstm_cell_118/BiasAddBiasAdd2sequential_89/lstm_113/while/lstm_cell_118/add:z:0Isequential_89/lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x|
:sequential_89/lstm_113/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_89/lstm_113/while/lstm_cell_118/splitSplitCsequential_89/lstm_113/while/lstm_cell_118/split/split_dim:output:0;sequential_89/lstm_113/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
2sequential_89/lstm_113/while/lstm_cell_118/SigmoidSigmoid9sequential_89/lstm_113/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_113/while/lstm_cell_118/Sigmoid_1Sigmoid9sequential_89/lstm_113/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
.sequential_89/lstm_113/while/lstm_cell_118/mulMul8sequential_89/lstm_113/while/lstm_cell_118/Sigmoid_1:y:0*sequential_89_lstm_113_while_placeholder_3*
T0*'
_output_shapes
:����������
/sequential_89/lstm_113/while/lstm_cell_118/ReluRelu9sequential_89/lstm_113/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
0sequential_89/lstm_113/while/lstm_cell_118/mul_1Mul6sequential_89/lstm_113/while/lstm_cell_118/Sigmoid:y:0=sequential_89/lstm_113/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
0sequential_89/lstm_113/while/lstm_cell_118/add_1AddV22sequential_89/lstm_113/while/lstm_cell_118/mul:z:04sequential_89/lstm_113/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_113/while/lstm_cell_118/Sigmoid_2Sigmoid9sequential_89/lstm_113/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:����������
1sequential_89/lstm_113/while/lstm_cell_118/Relu_1Relu4sequential_89/lstm_113/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
0sequential_89/lstm_113/while/lstm_cell_118/mul_2Mul8sequential_89/lstm_113/while/lstm_cell_118/Sigmoid_2:y:0?sequential_89/lstm_113/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:����������
Gsequential_89/lstm_113/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
Asequential_89/lstm_113/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_89_lstm_113_while_placeholder_1Psequential_89/lstm_113/while/TensorArrayV2Write/TensorListSetItem/index:output:04sequential_89/lstm_113/while/lstm_cell_118/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"sequential_89/lstm_113/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_89/lstm_113/while/addAddV2(sequential_89_lstm_113_while_placeholder+sequential_89/lstm_113/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_89/lstm_113/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"sequential_89/lstm_113/while/add_1AddV2Fsequential_89_lstm_113_while_sequential_89_lstm_113_while_loop_counter-sequential_89/lstm_113/while/add_1/y:output:0*
T0*
_output_shapes
: �
%sequential_89/lstm_113/while/IdentityIdentity&sequential_89/lstm_113/while/add_1:z:0"^sequential_89/lstm_113/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_113/while/Identity_1IdentityLsequential_89_lstm_113_while_sequential_89_lstm_113_while_maximum_iterations"^sequential_89/lstm_113/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_113/while/Identity_2Identity$sequential_89/lstm_113/while/add:z:0"^sequential_89/lstm_113/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_113/while/Identity_3IdentityQsequential_89/lstm_113/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_89/lstm_113/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_113/while/Identity_4Identity4sequential_89/lstm_113/while/lstm_cell_118/mul_2:z:0"^sequential_89/lstm_113/while/NoOp*
T0*'
_output_shapes
:����������
'sequential_89/lstm_113/while/Identity_5Identity4sequential_89/lstm_113/while/lstm_cell_118/add_1:z:0"^sequential_89/lstm_113/while/NoOp*
T0*'
_output_shapes
:����������
!sequential_89/lstm_113/while/NoOpNoOpB^sequential_89/lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOpA^sequential_89/lstm_113/while/lstm_cell_118/MatMul/ReadVariableOpC^sequential_89/lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_89_lstm_113_while_identity.sequential_89/lstm_113/while/Identity:output:0"[
'sequential_89_lstm_113_while_identity_10sequential_89/lstm_113/while/Identity_1:output:0"[
'sequential_89_lstm_113_while_identity_20sequential_89/lstm_113/while/Identity_2:output:0"[
'sequential_89_lstm_113_while_identity_30sequential_89/lstm_113/while/Identity_3:output:0"[
'sequential_89_lstm_113_while_identity_40sequential_89/lstm_113/while/Identity_4:output:0"[
'sequential_89_lstm_113_while_identity_50sequential_89/lstm_113/while/Identity_5:output:0"�
Jsequential_89_lstm_113_while_lstm_cell_118_biasadd_readvariableop_resourceLsequential_89_lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0"�
Ksequential_89_lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resourceMsequential_89_lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0"�
Isequential_89_lstm_113_while_lstm_cell_118_matmul_readvariableop_resourceKsequential_89_lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0"�
Csequential_89_lstm_113_while_sequential_89_lstm_113_strided_slice_1Esequential_89_lstm_113_while_sequential_89_lstm_113_strided_slice_1_0"�
sequential_89_lstm_113_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_113_tensorarrayunstack_tensorlistfromtensor�sequential_89_lstm_113_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_113_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2�
Asequential_89/lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOpAsequential_89/lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp2�
@sequential_89/lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp@sequential_89/lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp2�
Bsequential_89/lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOpBsequential_89/lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239891
lstm_111_input#
lstm_111_23239863:x#
lstm_111_23239865:x
lstm_111_23239867:x#
lstm_112_23239870:x#
lstm_112_23239872:x
lstm_112_23239874:x#
lstm_113_23239877:x#
lstm_113_23239879:x
lstm_113_23239881:x#
dense_87_23239885:
dense_87_23239887:
identity�� dense_87/StatefulPartitionedCall�"dropout_70/StatefulPartitionedCall� lstm_111/StatefulPartitionedCall� lstm_112/StatefulPartitionedCall� lstm_113/StatefulPartitionedCall�
 lstm_111/StatefulPartitionedCallStatefulPartitionedCalllstm_111_inputlstm_111_23239863lstm_111_23239865lstm_111_23239867*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23239708�
 lstm_112/StatefulPartitionedCallStatefulPartitionedCall)lstm_111/StatefulPartitionedCall:output:0lstm_112_23239870lstm_112_23239872lstm_112_23239874*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23239543�
 lstm_113/StatefulPartitionedCallStatefulPartitionedCall)lstm_112/StatefulPartitionedCall:output:0lstm_113_23239877lstm_113_23239879lstm_113_23239881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239378�
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall)lstm_113/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239217�
 dense_87/StatefulPartitionedCallStatefulPartitionedCall+dropout_70/StatefulPartitionedCall:output:0dense_87_23239885dense_87_23239887*
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
F__inference_dense_87_layer_call_and_return_conditional_losses_23239155x
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_87/StatefulPartitionedCall#^dropout_70/StatefulPartitionedCall!^lstm_111/StatefulPartitionedCall!^lstm_112/StatefulPartitionedCall!^lstm_113/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2D
 lstm_111/StatefulPartitionedCall lstm_111/StatefulPartitionedCall2D
 lstm_112/StatefulPartitionedCall lstm_112/StatefulPartitionedCall2D
 lstm_113/StatefulPartitionedCall lstm_113/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_111_input
�D
�

lstm_113_while_body_23240314.
*lstm_113_while_lstm_113_while_loop_counter4
0lstm_113_while_lstm_113_while_maximum_iterations
lstm_113_while_placeholder 
lstm_113_while_placeholder_1 
lstm_113_while_placeholder_2 
lstm_113_while_placeholder_3-
)lstm_113_while_lstm_113_strided_slice_1_0i
elstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0:xQ
?lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0:xL
>lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0:x
lstm_113_while_identity
lstm_113_while_identity_1
lstm_113_while_identity_2
lstm_113_while_identity_3
lstm_113_while_identity_4
lstm_113_while_identity_5+
'lstm_113_while_lstm_113_strided_slice_1g
clstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensorM
;lstm_113_while_lstm_cell_118_matmul_readvariableop_resource:xO
=lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource:xJ
<lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource:x��3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp�2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp�4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp�
@lstm_113/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_113/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensor_0lstm_113_while_placeholderIlstm_113/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp=lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_113/while/lstm_cell_118/MatMulMatMul9lstm_113/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp?lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_113/while/lstm_cell_118/MatMul_1MatMullstm_113_while_placeholder_2<lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_113/while/lstm_cell_118/addAddV2-lstm_113/while/lstm_cell_118/MatMul:product:0/lstm_113/while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp>lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_113/while/lstm_cell_118/BiasAddBiasAdd$lstm_113/while/lstm_cell_118/add:z:0;lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_113/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_113/while/lstm_cell_118/splitSplit5lstm_113/while/lstm_cell_118/split/split_dim:output:0-lstm_113/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_113/while/lstm_cell_118/SigmoidSigmoid+lstm_113/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:����������
&lstm_113/while/lstm_cell_118/Sigmoid_1Sigmoid+lstm_113/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
 lstm_113/while/lstm_cell_118/mulMul*lstm_113/while/lstm_cell_118/Sigmoid_1:y:0lstm_113_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_113/while/lstm_cell_118/ReluRelu+lstm_113/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
"lstm_113/while/lstm_cell_118/mul_1Mul(lstm_113/while/lstm_cell_118/Sigmoid:y:0/lstm_113/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_113/while/lstm_cell_118/add_1AddV2$lstm_113/while/lstm_cell_118/mul:z:0&lstm_113/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_113/while/lstm_cell_118/Sigmoid_2Sigmoid+lstm_113/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:����������
#lstm_113/while/lstm_cell_118/Relu_1Relu&lstm_113/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_113/while/lstm_cell_118/mul_2Mul*lstm_113/while/lstm_cell_118/Sigmoid_2:y:01lstm_113/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������{
9lstm_113/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
3lstm_113/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_113_while_placeholder_1Blstm_113/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_113/while/lstm_cell_118/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_113/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_113/while/addAddV2lstm_113_while_placeholderlstm_113/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_113/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/while/add_1AddV2*lstm_113_while_lstm_113_while_loop_counterlstm_113/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_113/while/IdentityIdentitylstm_113/while/add_1:z:0^lstm_113/while/NoOp*
T0*
_output_shapes
: �
lstm_113/while/Identity_1Identity0lstm_113_while_lstm_113_while_maximum_iterations^lstm_113/while/NoOp*
T0*
_output_shapes
: t
lstm_113/while/Identity_2Identitylstm_113/while/add:z:0^lstm_113/while/NoOp*
T0*
_output_shapes
: �
lstm_113/while/Identity_3IdentityClstm_113/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_113/while/NoOp*
T0*
_output_shapes
: �
lstm_113/while/Identity_4Identity&lstm_113/while/lstm_cell_118/mul_2:z:0^lstm_113/while/NoOp*
T0*'
_output_shapes
:����������
lstm_113/while/Identity_5Identity&lstm_113/while/lstm_cell_118/add_1:z:0^lstm_113/while/NoOp*
T0*'
_output_shapes
:����������
lstm_113/while/NoOpNoOp4^lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp3^lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp5^lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_113_while_identity lstm_113/while/Identity:output:0"?
lstm_113_while_identity_1"lstm_113/while/Identity_1:output:0"?
lstm_113_while_identity_2"lstm_113/while/Identity_2:output:0"?
lstm_113_while_identity_3"lstm_113/while/Identity_3:output:0"?
lstm_113_while_identity_4"lstm_113/while/Identity_4:output:0"?
lstm_113_while_identity_5"lstm_113/while/Identity_5:output:0"T
'lstm_113_while_lstm_113_strided_slice_1)lstm_113_while_lstm_113_strided_slice_1_0"~
<lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource>lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0"�
=lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource?lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0"|
;lstm_113_while_lstm_cell_118_matmul_readvariableop_resource=lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0"�
clstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensorelstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp2h
2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp2l
4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�

�
0__inference_sequential_89_layer_call_fn_23239187
lstm_111_input
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:x
	unknown_7:x
	unknown_8:
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_111_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239162o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_111_input
��
�
$__inference__traced_restore_23243306
file_prefix2
 assignvariableop_dense_87_kernel:.
 assignvariableop_1_dense_87_bias:B
0assignvariableop_2_lstm_111_lstm_cell_116_kernel:xL
:assignvariableop_3_lstm_111_lstm_cell_116_recurrent_kernel:x<
.assignvariableop_4_lstm_111_lstm_cell_116_bias:xB
0assignvariableop_5_lstm_112_lstm_cell_117_kernel:xL
:assignvariableop_6_lstm_112_lstm_cell_117_recurrent_kernel:x<
.assignvariableop_7_lstm_112_lstm_cell_117_bias:xB
0assignvariableop_8_lstm_113_lstm_cell_118_kernel:xL
:assignvariableop_9_lstm_113_lstm_cell_118_recurrent_kernel:x=
/assignvariableop_10_lstm_113_lstm_cell_118_bias:x'
assignvariableop_11_iteration:	 +
!assignvariableop_12_learning_rate: J
8assignvariableop_13_adam_m_lstm_111_lstm_cell_116_kernel:xJ
8assignvariableop_14_adam_v_lstm_111_lstm_cell_116_kernel:xT
Bassignvariableop_15_adam_m_lstm_111_lstm_cell_116_recurrent_kernel:xT
Bassignvariableop_16_adam_v_lstm_111_lstm_cell_116_recurrent_kernel:xD
6assignvariableop_17_adam_m_lstm_111_lstm_cell_116_bias:xD
6assignvariableop_18_adam_v_lstm_111_lstm_cell_116_bias:xJ
8assignvariableop_19_adam_m_lstm_112_lstm_cell_117_kernel:xJ
8assignvariableop_20_adam_v_lstm_112_lstm_cell_117_kernel:xT
Bassignvariableop_21_adam_m_lstm_112_lstm_cell_117_recurrent_kernel:xT
Bassignvariableop_22_adam_v_lstm_112_lstm_cell_117_recurrent_kernel:xD
6assignvariableop_23_adam_m_lstm_112_lstm_cell_117_bias:xD
6assignvariableop_24_adam_v_lstm_112_lstm_cell_117_bias:xJ
8assignvariableop_25_adam_m_lstm_113_lstm_cell_118_kernel:xJ
8assignvariableop_26_adam_v_lstm_113_lstm_cell_118_kernel:xT
Bassignvariableop_27_adam_m_lstm_113_lstm_cell_118_recurrent_kernel:xT
Bassignvariableop_28_adam_v_lstm_113_lstm_cell_118_recurrent_kernel:xD
6assignvariableop_29_adam_m_lstm_113_lstm_cell_118_bias:xD
6assignvariableop_30_adam_v_lstm_113_lstm_cell_118_bias:x<
*assignvariableop_31_adam_m_dense_87_kernel:<
*assignvariableop_32_adam_v_dense_87_kernel:6
(assignvariableop_33_adam_m_dense_87_bias:6
(assignvariableop_34_adam_v_dense_87_bias:%
assignvariableop_35_total_1: %
assignvariableop_36_count_1: #
assignvariableop_37_total: #
assignvariableop_38_count: 
identity_40��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_87_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_87_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_lstm_111_lstm_cell_116_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_lstm_111_lstm_cell_116_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_lstm_111_lstm_cell_116_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_lstm_112_lstm_cell_117_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp:assignvariableop_6_lstm_112_lstm_cell_117_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_112_lstm_cell_117_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_lstm_113_lstm_cell_118_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp:assignvariableop_9_lstm_113_lstm_cell_118_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_113_lstm_cell_118_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_iterationIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_learning_rateIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_m_lstm_111_lstm_cell_116_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp8assignvariableop_14_adam_v_lstm_111_lstm_cell_116_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpBassignvariableop_15_adam_m_lstm_111_lstm_cell_116_recurrent_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpBassignvariableop_16_adam_v_lstm_111_lstm_cell_116_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_m_lstm_111_lstm_cell_116_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_v_lstm_111_lstm_cell_116_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_m_lstm_112_lstm_cell_117_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_v_lstm_112_lstm_cell_117_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpBassignvariableop_21_adam_m_lstm_112_lstm_cell_117_recurrent_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adam_v_lstm_112_lstm_cell_117_recurrent_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_m_lstm_112_lstm_cell_117_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_v_lstm_112_lstm_cell_117_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_m_lstm_113_lstm_cell_118_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp8assignvariableop_26_adam_v_lstm_113_lstm_cell_118_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpBassignvariableop_27_adam_m_lstm_113_lstm_cell_118_recurrent_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_v_lstm_113_lstm_cell_118_recurrent_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_m_lstm_113_lstm_cell_118_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_v_lstm_113_lstm_cell_118_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_m_dense_87_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_v_dense_87_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_m_dense_87_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_v_dense_87_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
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
while_cond_23238599
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23238599___redundant_placeholder06
2while_while_cond_23238599___redundant_placeholder16
2while_while_cond_23238599___redundant_placeholder26
2while_while_cond_23238599___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�8
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238125

inputs(
lstm_cell_117_23238043:x(
lstm_cell_117_23238045:x$
lstm_cell_117_23238047:x
identity��%lstm_cell_117/StatefulPartitionedCall�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
%lstm_cell_117/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_117_23238043lstm_cell_117_23238045lstm_cell_117_23238047*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238042n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_117_23238043lstm_cell_117_23238045lstm_cell_117_23238047*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23238056*
condR
while_cond_23238055*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������v
NoOpNoOp&^lstm_cell_117/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_117/StatefulPartitionedCall%lstm_cell_117/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
+__inference_lstm_113_layer_call_fn_23242108

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239130o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238188

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
й
�
#__inference__wrapped_model_23237625
lstm_111_inputU
Csequential_89_lstm_111_lstm_cell_116_matmul_readvariableop_resource:xW
Esequential_89_lstm_111_lstm_cell_116_matmul_1_readvariableop_resource:xR
Dsequential_89_lstm_111_lstm_cell_116_biasadd_readvariableop_resource:xU
Csequential_89_lstm_112_lstm_cell_117_matmul_readvariableop_resource:xW
Esequential_89_lstm_112_lstm_cell_117_matmul_1_readvariableop_resource:xR
Dsequential_89_lstm_112_lstm_cell_117_biasadd_readvariableop_resource:xU
Csequential_89_lstm_113_lstm_cell_118_matmul_readvariableop_resource:xW
Esequential_89_lstm_113_lstm_cell_118_matmul_1_readvariableop_resource:xR
Dsequential_89_lstm_113_lstm_cell_118_biasadd_readvariableop_resource:xG
5sequential_89_dense_87_matmul_readvariableop_resource:D
6sequential_89_dense_87_biasadd_readvariableop_resource:
identity��-sequential_89/dense_87/BiasAdd/ReadVariableOp�,sequential_89/dense_87/MatMul/ReadVariableOp�;sequential_89/lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp�:sequential_89/lstm_111/lstm_cell_116/MatMul/ReadVariableOp�<sequential_89/lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp�sequential_89/lstm_111/while�;sequential_89/lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp�:sequential_89/lstm_112/lstm_cell_117/MatMul/ReadVariableOp�<sequential_89/lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp�sequential_89/lstm_112/while�;sequential_89/lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp�:sequential_89/lstm_113/lstm_cell_118/MatMul/ReadVariableOp�<sequential_89/lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp�sequential_89/lstm_113/whileZ
sequential_89/lstm_111/ShapeShapelstm_111_input*
T0*
_output_shapes
:t
*sequential_89/lstm_111/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_89/lstm_111/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_89/lstm_111/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_89/lstm_111/strided_sliceStridedSlice%sequential_89/lstm_111/Shape:output:03sequential_89/lstm_111/strided_slice/stack:output:05sequential_89/lstm_111/strided_slice/stack_1:output:05sequential_89/lstm_111/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_89/lstm_111/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
#sequential_89/lstm_111/zeros/packedPack-sequential_89/lstm_111/strided_slice:output:0.sequential_89/lstm_111/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_89/lstm_111/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_89/lstm_111/zerosFill,sequential_89/lstm_111/zeros/packed:output:0+sequential_89/lstm_111/zeros/Const:output:0*
T0*'
_output_shapes
:���������i
'sequential_89/lstm_111/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%sequential_89/lstm_111/zeros_1/packedPack-sequential_89/lstm_111/strided_slice:output:00sequential_89/lstm_111/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_89/lstm_111/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_89/lstm_111/zeros_1Fill.sequential_89/lstm_111/zeros_1/packed:output:0-sequential_89/lstm_111/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������z
%sequential_89/lstm_111/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_89/lstm_111/transpose	Transposelstm_111_input.sequential_89/lstm_111/transpose/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_89/lstm_111/Shape_1Shape$sequential_89/lstm_111/transpose:y:0*
T0*
_output_shapes
:v
,sequential_89/lstm_111/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_111/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_89/lstm_111/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_111/strided_slice_1StridedSlice'sequential_89/lstm_111/Shape_1:output:05sequential_89/lstm_111/strided_slice_1/stack:output:07sequential_89/lstm_111/strided_slice_1/stack_1:output:07sequential_89/lstm_111/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_89/lstm_111/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_89/lstm_111/TensorArrayV2TensorListReserve;sequential_89/lstm_111/TensorArrayV2/element_shape:output:0/sequential_89/lstm_111/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lsequential_89/lstm_111/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_89/lstm_111/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_89/lstm_111/transpose:y:0Usequential_89/lstm_111/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,sequential_89/lstm_111/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_111/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_89/lstm_111/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_111/strided_slice_2StridedSlice$sequential_89/lstm_111/transpose:y:05sequential_89/lstm_111/strided_slice_2/stack:output:07sequential_89/lstm_111/strided_slice_2/stack_1:output:07sequential_89/lstm_111/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
:sequential_89/lstm_111/lstm_cell_116/MatMul/ReadVariableOpReadVariableOpCsequential_89_lstm_111_lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
+sequential_89/lstm_111/lstm_cell_116/MatMulMatMul/sequential_89/lstm_111/strided_slice_2:output:0Bsequential_89/lstm_111/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
<sequential_89/lstm_111/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOpEsequential_89_lstm_111_lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
-sequential_89/lstm_111/lstm_cell_116/MatMul_1MatMul%sequential_89/lstm_111/zeros:output:0Dsequential_89/lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
(sequential_89/lstm_111/lstm_cell_116/addAddV25sequential_89/lstm_111/lstm_cell_116/MatMul:product:07sequential_89/lstm_111/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
;sequential_89/lstm_111/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOpDsequential_89_lstm_111_lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
,sequential_89/lstm_111/lstm_cell_116/BiasAddBiasAdd,sequential_89/lstm_111/lstm_cell_116/add:z:0Csequential_89/lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xv
4sequential_89/lstm_111/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_89/lstm_111/lstm_cell_116/splitSplit=sequential_89/lstm_111/lstm_cell_116/split/split_dim:output:05sequential_89/lstm_111/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
,sequential_89/lstm_111/lstm_cell_116/SigmoidSigmoid3sequential_89/lstm_111/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:����������
.sequential_89/lstm_111/lstm_cell_116/Sigmoid_1Sigmoid3sequential_89/lstm_111/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
(sequential_89/lstm_111/lstm_cell_116/mulMul2sequential_89/lstm_111/lstm_cell_116/Sigmoid_1:y:0'sequential_89/lstm_111/zeros_1:output:0*
T0*'
_output_shapes
:����������
)sequential_89/lstm_111/lstm_cell_116/ReluRelu3sequential_89/lstm_111/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
*sequential_89/lstm_111/lstm_cell_116/mul_1Mul0sequential_89/lstm_111/lstm_cell_116/Sigmoid:y:07sequential_89/lstm_111/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
*sequential_89/lstm_111/lstm_cell_116/add_1AddV2,sequential_89/lstm_111/lstm_cell_116/mul:z:0.sequential_89/lstm_111/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:����������
.sequential_89/lstm_111/lstm_cell_116/Sigmoid_2Sigmoid3sequential_89/lstm_111/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:����������
+sequential_89/lstm_111/lstm_cell_116/Relu_1Relu.sequential_89/lstm_111/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
*sequential_89/lstm_111/lstm_cell_116/mul_2Mul2sequential_89/lstm_111/lstm_cell_116/Sigmoid_2:y:09sequential_89/lstm_111/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_111/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
&sequential_89/lstm_111/TensorArrayV2_1TensorListReserve=sequential_89/lstm_111/TensorArrayV2_1/element_shape:output:0/sequential_89/lstm_111/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
sequential_89/lstm_111/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_89/lstm_111/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������k
)sequential_89/lstm_111/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_89/lstm_111/whileWhile2sequential_89/lstm_111/while/loop_counter:output:08sequential_89/lstm_111/while/maximum_iterations:output:0$sequential_89/lstm_111/time:output:0/sequential_89/lstm_111/TensorArrayV2_1:handle:0%sequential_89/lstm_111/zeros:output:0'sequential_89/lstm_111/zeros_1:output:0/sequential_89/lstm_111/strided_slice_1:output:0Nsequential_89/lstm_111/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_89_lstm_111_lstm_cell_116_matmul_readvariableop_resourceEsequential_89_lstm_111_lstm_cell_116_matmul_1_readvariableop_resourceDsequential_89_lstm_111_lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_89_lstm_111_while_body_23237254*6
cond.R,
*sequential_89_lstm_111_while_cond_23237253*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Gsequential_89/lstm_111/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
9sequential_89/lstm_111/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_89/lstm_111/while:output:3Psequential_89/lstm_111/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0
,sequential_89/lstm_111/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.sequential_89/lstm_111/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_111/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_111/strided_slice_3StridedSliceBsequential_89/lstm_111/TensorArrayV2Stack/TensorListStack:tensor:05sequential_89/lstm_111/strided_slice_3/stack:output:07sequential_89/lstm_111/strided_slice_3/stack_1:output:07sequential_89/lstm_111/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask|
'sequential_89/lstm_111/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"sequential_89/lstm_111/transpose_1	TransposeBsequential_89/lstm_111/TensorArrayV2Stack/TensorListStack:tensor:00sequential_89/lstm_111/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_89/lstm_111/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    r
sequential_89/lstm_112/ShapeShape&sequential_89/lstm_111/transpose_1:y:0*
T0*
_output_shapes
:t
*sequential_89/lstm_112/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_89/lstm_112/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_89/lstm_112/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_89/lstm_112/strided_sliceStridedSlice%sequential_89/lstm_112/Shape:output:03sequential_89/lstm_112/strided_slice/stack:output:05sequential_89/lstm_112/strided_slice/stack_1:output:05sequential_89/lstm_112/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_89/lstm_112/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
#sequential_89/lstm_112/zeros/packedPack-sequential_89/lstm_112/strided_slice:output:0.sequential_89/lstm_112/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_89/lstm_112/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_89/lstm_112/zerosFill,sequential_89/lstm_112/zeros/packed:output:0+sequential_89/lstm_112/zeros/Const:output:0*
T0*'
_output_shapes
:���������i
'sequential_89/lstm_112/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%sequential_89/lstm_112/zeros_1/packedPack-sequential_89/lstm_112/strided_slice:output:00sequential_89/lstm_112/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_89/lstm_112/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_89/lstm_112/zeros_1Fill.sequential_89/lstm_112/zeros_1/packed:output:0-sequential_89/lstm_112/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������z
%sequential_89/lstm_112/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_89/lstm_112/transpose	Transpose&sequential_89/lstm_111/transpose_1:y:0.sequential_89/lstm_112/transpose/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_89/lstm_112/Shape_1Shape$sequential_89/lstm_112/transpose:y:0*
T0*
_output_shapes
:v
,sequential_89/lstm_112/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_112/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_89/lstm_112/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_112/strided_slice_1StridedSlice'sequential_89/lstm_112/Shape_1:output:05sequential_89/lstm_112/strided_slice_1/stack:output:07sequential_89/lstm_112/strided_slice_1/stack_1:output:07sequential_89/lstm_112/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_89/lstm_112/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_89/lstm_112/TensorArrayV2TensorListReserve;sequential_89/lstm_112/TensorArrayV2/element_shape:output:0/sequential_89/lstm_112/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lsequential_89/lstm_112/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_89/lstm_112/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_89/lstm_112/transpose:y:0Usequential_89/lstm_112/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,sequential_89/lstm_112/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_112/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_89/lstm_112/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_112/strided_slice_2StridedSlice$sequential_89/lstm_112/transpose:y:05sequential_89/lstm_112/strided_slice_2/stack:output:07sequential_89/lstm_112/strided_slice_2/stack_1:output:07sequential_89/lstm_112/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
:sequential_89/lstm_112/lstm_cell_117/MatMul/ReadVariableOpReadVariableOpCsequential_89_lstm_112_lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
+sequential_89/lstm_112/lstm_cell_117/MatMulMatMul/sequential_89/lstm_112/strided_slice_2:output:0Bsequential_89/lstm_112/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
<sequential_89/lstm_112/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOpEsequential_89_lstm_112_lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
-sequential_89/lstm_112/lstm_cell_117/MatMul_1MatMul%sequential_89/lstm_112/zeros:output:0Dsequential_89/lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
(sequential_89/lstm_112/lstm_cell_117/addAddV25sequential_89/lstm_112/lstm_cell_117/MatMul:product:07sequential_89/lstm_112/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
;sequential_89/lstm_112/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOpDsequential_89_lstm_112_lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
,sequential_89/lstm_112/lstm_cell_117/BiasAddBiasAdd,sequential_89/lstm_112/lstm_cell_117/add:z:0Csequential_89/lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xv
4sequential_89/lstm_112/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_89/lstm_112/lstm_cell_117/splitSplit=sequential_89/lstm_112/lstm_cell_117/split/split_dim:output:05sequential_89/lstm_112/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
,sequential_89/lstm_112/lstm_cell_117/SigmoidSigmoid3sequential_89/lstm_112/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:����������
.sequential_89/lstm_112/lstm_cell_117/Sigmoid_1Sigmoid3sequential_89/lstm_112/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
(sequential_89/lstm_112/lstm_cell_117/mulMul2sequential_89/lstm_112/lstm_cell_117/Sigmoid_1:y:0'sequential_89/lstm_112/zeros_1:output:0*
T0*'
_output_shapes
:����������
)sequential_89/lstm_112/lstm_cell_117/ReluRelu3sequential_89/lstm_112/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
*sequential_89/lstm_112/lstm_cell_117/mul_1Mul0sequential_89/lstm_112/lstm_cell_117/Sigmoid:y:07sequential_89/lstm_112/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
*sequential_89/lstm_112/lstm_cell_117/add_1AddV2,sequential_89/lstm_112/lstm_cell_117/mul:z:0.sequential_89/lstm_112/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:����������
.sequential_89/lstm_112/lstm_cell_117/Sigmoid_2Sigmoid3sequential_89/lstm_112/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:����������
+sequential_89/lstm_112/lstm_cell_117/Relu_1Relu.sequential_89/lstm_112/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
*sequential_89/lstm_112/lstm_cell_117/mul_2Mul2sequential_89/lstm_112/lstm_cell_117/Sigmoid_2:y:09sequential_89/lstm_112/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_112/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
&sequential_89/lstm_112/TensorArrayV2_1TensorListReserve=sequential_89/lstm_112/TensorArrayV2_1/element_shape:output:0/sequential_89/lstm_112/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
sequential_89/lstm_112/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_89/lstm_112/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������k
)sequential_89/lstm_112/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_89/lstm_112/whileWhile2sequential_89/lstm_112/while/loop_counter:output:08sequential_89/lstm_112/while/maximum_iterations:output:0$sequential_89/lstm_112/time:output:0/sequential_89/lstm_112/TensorArrayV2_1:handle:0%sequential_89/lstm_112/zeros:output:0'sequential_89/lstm_112/zeros_1:output:0/sequential_89/lstm_112/strided_slice_1:output:0Nsequential_89/lstm_112/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_89_lstm_112_lstm_cell_117_matmul_readvariableop_resourceEsequential_89_lstm_112_lstm_cell_117_matmul_1_readvariableop_resourceDsequential_89_lstm_112_lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_89_lstm_112_while_body_23237393*6
cond.R,
*sequential_89_lstm_112_while_cond_23237392*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Gsequential_89/lstm_112/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
9sequential_89/lstm_112/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_89/lstm_112/while:output:3Psequential_89/lstm_112/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0
,sequential_89/lstm_112/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.sequential_89/lstm_112/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_112/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_112/strided_slice_3StridedSliceBsequential_89/lstm_112/TensorArrayV2Stack/TensorListStack:tensor:05sequential_89/lstm_112/strided_slice_3/stack:output:07sequential_89/lstm_112/strided_slice_3/stack_1:output:07sequential_89/lstm_112/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask|
'sequential_89/lstm_112/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"sequential_89/lstm_112/transpose_1	TransposeBsequential_89/lstm_112/TensorArrayV2Stack/TensorListStack:tensor:00sequential_89/lstm_112/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_89/lstm_112/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    r
sequential_89/lstm_113/ShapeShape&sequential_89/lstm_112/transpose_1:y:0*
T0*
_output_shapes
:t
*sequential_89/lstm_113/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_89/lstm_113/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_89/lstm_113/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_89/lstm_113/strided_sliceStridedSlice%sequential_89/lstm_113/Shape:output:03sequential_89/lstm_113/strided_slice/stack:output:05sequential_89/lstm_113/strided_slice/stack_1:output:05sequential_89/lstm_113/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_89/lstm_113/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
#sequential_89/lstm_113/zeros/packedPack-sequential_89/lstm_113/strided_slice:output:0.sequential_89/lstm_113/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_89/lstm_113/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_89/lstm_113/zerosFill,sequential_89/lstm_113/zeros/packed:output:0+sequential_89/lstm_113/zeros/Const:output:0*
T0*'
_output_shapes
:���������i
'sequential_89/lstm_113/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%sequential_89/lstm_113/zeros_1/packedPack-sequential_89/lstm_113/strided_slice:output:00sequential_89/lstm_113/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_89/lstm_113/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_89/lstm_113/zeros_1Fill.sequential_89/lstm_113/zeros_1/packed:output:0-sequential_89/lstm_113/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������z
%sequential_89/lstm_113/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_89/lstm_113/transpose	Transpose&sequential_89/lstm_112/transpose_1:y:0.sequential_89/lstm_113/transpose/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_89/lstm_113/Shape_1Shape$sequential_89/lstm_113/transpose:y:0*
T0*
_output_shapes
:v
,sequential_89/lstm_113/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_113/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_89/lstm_113/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_113/strided_slice_1StridedSlice'sequential_89/lstm_113/Shape_1:output:05sequential_89/lstm_113/strided_slice_1/stack:output:07sequential_89/lstm_113/strided_slice_1/stack_1:output:07sequential_89/lstm_113/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_89/lstm_113/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_89/lstm_113/TensorArrayV2TensorListReserve;sequential_89/lstm_113/TensorArrayV2/element_shape:output:0/sequential_89/lstm_113/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lsequential_89/lstm_113/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_89/lstm_113/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_89/lstm_113/transpose:y:0Usequential_89/lstm_113/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,sequential_89/lstm_113/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_113/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_89/lstm_113/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_113/strided_slice_2StridedSlice$sequential_89/lstm_113/transpose:y:05sequential_89/lstm_113/strided_slice_2/stack:output:07sequential_89/lstm_113/strided_slice_2/stack_1:output:07sequential_89/lstm_113/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
:sequential_89/lstm_113/lstm_cell_118/MatMul/ReadVariableOpReadVariableOpCsequential_89_lstm_113_lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
+sequential_89/lstm_113/lstm_cell_118/MatMulMatMul/sequential_89/lstm_113/strided_slice_2:output:0Bsequential_89/lstm_113/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
<sequential_89/lstm_113/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOpEsequential_89_lstm_113_lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
-sequential_89/lstm_113/lstm_cell_118/MatMul_1MatMul%sequential_89/lstm_113/zeros:output:0Dsequential_89/lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
(sequential_89/lstm_113/lstm_cell_118/addAddV25sequential_89/lstm_113/lstm_cell_118/MatMul:product:07sequential_89/lstm_113/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
;sequential_89/lstm_113/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOpDsequential_89_lstm_113_lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
,sequential_89/lstm_113/lstm_cell_118/BiasAddBiasAdd,sequential_89/lstm_113/lstm_cell_118/add:z:0Csequential_89/lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xv
4sequential_89/lstm_113/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_89/lstm_113/lstm_cell_118/splitSplit=sequential_89/lstm_113/lstm_cell_118/split/split_dim:output:05sequential_89/lstm_113/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
,sequential_89/lstm_113/lstm_cell_118/SigmoidSigmoid3sequential_89/lstm_113/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:����������
.sequential_89/lstm_113/lstm_cell_118/Sigmoid_1Sigmoid3sequential_89/lstm_113/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
(sequential_89/lstm_113/lstm_cell_118/mulMul2sequential_89/lstm_113/lstm_cell_118/Sigmoid_1:y:0'sequential_89/lstm_113/zeros_1:output:0*
T0*'
_output_shapes
:����������
)sequential_89/lstm_113/lstm_cell_118/ReluRelu3sequential_89/lstm_113/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
*sequential_89/lstm_113/lstm_cell_118/mul_1Mul0sequential_89/lstm_113/lstm_cell_118/Sigmoid:y:07sequential_89/lstm_113/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
*sequential_89/lstm_113/lstm_cell_118/add_1AddV2,sequential_89/lstm_113/lstm_cell_118/mul:z:0.sequential_89/lstm_113/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:����������
.sequential_89/lstm_113/lstm_cell_118/Sigmoid_2Sigmoid3sequential_89/lstm_113/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:����������
+sequential_89/lstm_113/lstm_cell_118/Relu_1Relu.sequential_89/lstm_113/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
*sequential_89/lstm_113/lstm_cell_118/mul_2Mul2sequential_89/lstm_113/lstm_cell_118/Sigmoid_2:y:09sequential_89/lstm_113/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_113/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   u
3sequential_89/lstm_113/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
&sequential_89/lstm_113/TensorArrayV2_1TensorListReserve=sequential_89/lstm_113/TensorArrayV2_1/element_shape:output:0<sequential_89/lstm_113/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
sequential_89/lstm_113/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_89/lstm_113/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������k
)sequential_89/lstm_113/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_89/lstm_113/whileWhile2sequential_89/lstm_113/while/loop_counter:output:08sequential_89/lstm_113/while/maximum_iterations:output:0$sequential_89/lstm_113/time:output:0/sequential_89/lstm_113/TensorArrayV2_1:handle:0%sequential_89/lstm_113/zeros:output:0'sequential_89/lstm_113/zeros_1:output:0/sequential_89/lstm_113/strided_slice_1:output:0Nsequential_89/lstm_113/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_89_lstm_113_lstm_cell_118_matmul_readvariableop_resourceEsequential_89_lstm_113_lstm_cell_118_matmul_1_readvariableop_resourceDsequential_89_lstm_113_lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_89_lstm_113_while_body_23237533*6
cond.R,
*sequential_89_lstm_113_while_cond_23237532*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Gsequential_89/lstm_113/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
9sequential_89/lstm_113/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_89/lstm_113/while:output:3Psequential_89/lstm_113/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elements
,sequential_89/lstm_113/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.sequential_89/lstm_113/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_89/lstm_113/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_89/lstm_113/strided_slice_3StridedSliceBsequential_89/lstm_113/TensorArrayV2Stack/TensorListStack:tensor:05sequential_89/lstm_113/strided_slice_3/stack:output:07sequential_89/lstm_113/strided_slice_3/stack_1:output:07sequential_89/lstm_113/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask|
'sequential_89/lstm_113/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"sequential_89/lstm_113/transpose_1	TransposeBsequential_89/lstm_113/TensorArrayV2Stack/TensorListStack:tensor:00sequential_89/lstm_113/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_89/lstm_113/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
!sequential_89/dropout_70/IdentityIdentity/sequential_89/lstm_113/strided_slice_3:output:0*
T0*'
_output_shapes
:����������
,sequential_89/dense_87/MatMul/ReadVariableOpReadVariableOp5sequential_89_dense_87_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_89/dense_87/MatMulMatMul*sequential_89/dropout_70/Identity:output:04sequential_89/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_89/dense_87/BiasAdd/ReadVariableOpReadVariableOp6sequential_89_dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_89/dense_87/BiasAddBiasAdd'sequential_89/dense_87/MatMul:product:05sequential_89/dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_89/dense_87/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_89/dense_87/BiasAdd/ReadVariableOp-^sequential_89/dense_87/MatMul/ReadVariableOp<^sequential_89/lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp;^sequential_89/lstm_111/lstm_cell_116/MatMul/ReadVariableOp=^sequential_89/lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp^sequential_89/lstm_111/while<^sequential_89/lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp;^sequential_89/lstm_112/lstm_cell_117/MatMul/ReadVariableOp=^sequential_89/lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp^sequential_89/lstm_112/while<^sequential_89/lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp;^sequential_89/lstm_113/lstm_cell_118/MatMul/ReadVariableOp=^sequential_89/lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp^sequential_89/lstm_113/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2^
-sequential_89/dense_87/BiasAdd/ReadVariableOp-sequential_89/dense_87/BiasAdd/ReadVariableOp2\
,sequential_89/dense_87/MatMul/ReadVariableOp,sequential_89/dense_87/MatMul/ReadVariableOp2z
;sequential_89/lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp;sequential_89/lstm_111/lstm_cell_116/BiasAdd/ReadVariableOp2x
:sequential_89/lstm_111/lstm_cell_116/MatMul/ReadVariableOp:sequential_89/lstm_111/lstm_cell_116/MatMul/ReadVariableOp2|
<sequential_89/lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp<sequential_89/lstm_111/lstm_cell_116/MatMul_1/ReadVariableOp2<
sequential_89/lstm_111/whilesequential_89/lstm_111/while2z
;sequential_89/lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp;sequential_89/lstm_112/lstm_cell_117/BiasAdd/ReadVariableOp2x
:sequential_89/lstm_112/lstm_cell_117/MatMul/ReadVariableOp:sequential_89/lstm_112/lstm_cell_117/MatMul/ReadVariableOp2|
<sequential_89/lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp<sequential_89/lstm_112/lstm_cell_117/MatMul_1/ReadVariableOp2<
sequential_89/lstm_112/whilesequential_89/lstm_112/while2z
;sequential_89/lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp;sequential_89/lstm_113/lstm_cell_118/BiasAdd/ReadVariableOp2x
:sequential_89/lstm_113/lstm_cell_118/MatMul/ReadVariableOp:sequential_89/lstm_113/lstm_cell_118/MatMul/ReadVariableOp2|
<sequential_89/lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp<sequential_89/lstm_113/lstm_cell_118/MatMul_1/ReadVariableOp2<
sequential_89/lstm_113/whilesequential_89/lstm_113/while:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_111_input
�
�
0__inference_lstm_cell_116_layer_call_fn_23242779

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237838o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�
�
while_cond_23241704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23241704___redundant_placeholder06
2while_while_cond_23241704___redundant_placeholder16
2while_while_cond_23241704___redundant_placeholder26
2while_while_cond_23241704___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�8
�
while_body_23240946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_116_matmul_readvariableop_resource_0:xH
6while_lstm_cell_116_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_116_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_116_matmul_readvariableop_resource:xF
4while_lstm_cell_116_matmul_1_readvariableop_resource:xA
3while_lstm_cell_116_biasadd_readvariableop_resource:x��*while/lstm_cell_116/BiasAdd/ReadVariableOp�)while/lstm_cell_116/MatMul/ReadVariableOp�+while/lstm_cell_116/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_116/addAddV2$while/lstm_cell_116/MatMul:product:0&while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_116/BiasAddBiasAddwhile/lstm_cell_116/add:z:02while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:0$while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_116/SigmoidSigmoid"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_1Sigmoid"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mulMul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_116/ReluRelu"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_1Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/add_1AddV2while/lstm_cell_116/mul:z:0while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_2Sigmoid"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_2Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_116/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_116/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_116/BiasAdd/ReadVariableOp*^while/lstm_cell_116/MatMul/ReadVariableOp,^while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_116_biasadd_readvariableop_resource5while_lstm_cell_116_biasadd_readvariableop_resource_0"n
4while_lstm_cell_116_matmul_1_readvariableop_resource6while_lstm_cell_116_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_116_matmul_readvariableop_resource4while_lstm_cell_116_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_116/BiasAdd/ReadVariableOp*while/lstm_cell_116/BiasAdd/ReadVariableOp2V
)while/lstm_cell_116/MatMul/ReadVariableOp)while/lstm_cell_116/MatMul/ReadVariableOp2Z
+while/lstm_cell_116/MatMul_1/ReadVariableOp+while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�#
�
while_body_23238056
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_117_23238080_0:x0
while_lstm_cell_117_23238082_0:x,
while_lstm_cell_117_23238084_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_117_23238080:x.
while_lstm_cell_117_23238082:x*
while_lstm_cell_117_23238084:x��+while/lstm_cell_117/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_117/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_117_23238080_0while_lstm_cell_117_23238082_0while_lstm_cell_117_23238084_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238042�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_117/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_117/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_117/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_117/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_117_23238080while_lstm_cell_117_23238080_0">
while_lstm_cell_117_23238082while_lstm_cell_117_23238082_0">
while_lstm_cell_117_23238084while_lstm_cell_117_23238084_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_117/StatefulPartitionedCall+while/lstm_cell_117/StatefulPartitionedCall: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�9
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23238477

inputs(
lstm_cell_118_23238393:x(
lstm_cell_118_23238395:x$
lstm_cell_118_23238397:x
identity��%lstm_cell_118/StatefulPartitionedCall�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
%lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_118_23238393lstm_cell_118_23238395lstm_cell_118_23238397*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238392n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_118_23238393lstm_cell_118_23238395lstm_cell_118_23238397*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23238407*
condR
while_cond_23238406*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������v
NoOpNoOp&^lstm_cell_118/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_118/StatefulPartitionedCall%lstm_cell_118/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�J
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241316

inputs>
,lstm_cell_116_matmul_readvariableop_resource:x@
.lstm_cell_116_matmul_1_readvariableop_resource:x;
-lstm_cell_116_biasadd_readvariableop_resource:x
identity��$lstm_cell_116/BiasAdd/ReadVariableOp�#lstm_cell_116/MatMul/ReadVariableOp�%lstm_cell_116/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_116/MatMul/ReadVariableOpReadVariableOp,lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0+lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMul_1MatMulzeros:output:0-lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_116/addAddV2lstm_cell_116/MatMul:product:0 lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_116/BiasAddBiasAddlstm_cell_116/add:z:0,lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_116/SigmoidSigmoidlstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_116/mulMullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_116/ReluRelulstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_1Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_116/add_1AddV2lstm_cell_116/mul:z:0lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_116/Relu_1Relulstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_2Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_116_matmul_readvariableop_resource.lstm_cell_116_matmul_1_readvariableop_resource-lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23241232*
condR
while_cond_23241231*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_116/BiasAdd/ReadVariableOp$^lstm_cell_116/MatMul/ReadVariableOp&^lstm_cell_116/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_116/BiasAdd/ReadVariableOp$lstm_cell_116/BiasAdd/ReadVariableOp2J
#lstm_cell_116/MatMul/ReadVariableOp#lstm_cell_116/MatMul/ReadVariableOp2N
%lstm_cell_116/MatMul_1/ReadVariableOp%lstm_cell_116/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242554

inputs>
,lstm_cell_118_matmul_readvariableop_resource:x@
.lstm_cell_118_matmul_1_readvariableop_resource:x;
-lstm_cell_118_biasadd_readvariableop_resource:x
identity��$lstm_cell_118/BiasAdd/ReadVariableOp�#lstm_cell_118/MatMul/ReadVariableOp�%lstm_cell_118/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23242469*
condR
while_cond_23242468*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_23239045
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_118_matmul_readvariableop_resource_0:xH
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_118_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_118_matmul_readvariableop_resource:xF
4while_lstm_cell_118_matmul_1_readvariableop_resource:xA
3while_lstm_cell_118_biasadd_readvariableop_resource:x��*while/lstm_cell_118/BiasAdd/ReadVariableOp�)while/lstm_cell_118/MatMul/ReadVariableOp�+while/lstm_cell_118/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_118/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
f
-__inference_dropout_70_layer_call_fn_23242709

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239217o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_lstm_113_layer_call_fn_23242119

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239378o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�L
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242264
inputs_0>
,lstm_cell_118_matmul_readvariableop_resource:x@
.lstm_cell_118_matmul_1_readvariableop_resource:x;
-lstm_cell_118_biasadd_readvariableop_resource:x
identity��$lstm_cell_118/BiasAdd/ReadVariableOp�#lstm_cell_118/MatMul/ReadVariableOp�%lstm_cell_118/MatMul_1/ReadVariableOp�while=
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23242179*
condR
while_cond_23242178*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�9
�
while_body_23242469
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_118_matmul_readvariableop_resource_0:xH
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_118_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_118_matmul_readvariableop_resource:xF
4while_lstm_cell_118_matmul_1_readvariableop_resource:xA
3while_lstm_cell_118_biasadd_readvariableop_resource:x��*while/lstm_cell_118/BiasAdd/ReadVariableOp�)while/lstm_cell_118/MatMul/ReadVariableOp�+while/lstm_cell_118/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_118/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_23241991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_117_matmul_readvariableop_resource_0:xH
6while_lstm_cell_117_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_117_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_117_matmul_readvariableop_resource:xF
4while_lstm_cell_117_matmul_1_readvariableop_resource:xA
3while_lstm_cell_117_biasadd_readvariableop_resource:x��*while/lstm_cell_117/BiasAdd/ReadVariableOp�)while/lstm_cell_117/MatMul/ReadVariableOp�+while/lstm_cell_117/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_117/addAddV2$while/lstm_cell_117/MatMul:product:0&while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_117/BiasAddBiasAddwhile/lstm_cell_117/add:z:02while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_117/splitSplit,while/lstm_cell_117/split/split_dim:output:0$while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_117/SigmoidSigmoid"while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_1Sigmoid"while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mulMul!while/lstm_cell_117/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_117/ReluRelu"while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_1Mulwhile/lstm_cell_117/Sigmoid:y:0&while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/add_1AddV2while/lstm_cell_117/mul:z:0while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_2Sigmoid"while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_117/Relu_1Reluwhile/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_2Mul!while/lstm_cell_117/Sigmoid_2:y:0(while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_117/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_117/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_117/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_117/BiasAdd/ReadVariableOp*^while/lstm_cell_117/MatMul/ReadVariableOp,^while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_117_biasadd_readvariableop_resource5while_lstm_cell_117_biasadd_readvariableop_resource_0"n
4while_lstm_cell_117_matmul_1_readvariableop_resource6while_lstm_cell_117_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_117_matmul_readvariableop_resource4while_lstm_cell_117_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_117/BiasAdd/ReadVariableOp*while/lstm_cell_117/BiasAdd/ReadVariableOp2V
)while/lstm_cell_117/MatMul/ReadVariableOp)while/lstm_cell_117/MatMul/ReadVariableOp2Z
+while/lstm_cell_117/MatMul_1/ReadVariableOp+while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
f
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239143

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23239130

inputs>
,lstm_cell_118_matmul_readvariableop_resource:x@
.lstm_cell_118_matmul_1_readvariableop_resource:x;
-lstm_cell_118_biasadd_readvariableop_resource:x
identity��$lstm_cell_118/BiasAdd/ReadVariableOp�#lstm_cell_118/MatMul/ReadVariableOp�%lstm_cell_118/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23239045*
condR
while_cond_23239044*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_23241088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23241088___redundant_placeholder06
2while_while_cond_23241088___redundant_placeholder16
2while_while_cond_23241088___redundant_placeholder26
2while_while_cond_23241088___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242909

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�
�
while_cond_23237705
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23237705___redundant_placeholder06
2while_while_cond_23237705___redundant_placeholder16
2while_while_cond_23237705___redundant_placeholder26
2while_while_cond_23237705___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�

�
0__inference_sequential_89_layer_call_fn_23239829
lstm_111_input
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:x
	unknown_7:x
	unknown_8:
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_111_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_111_input
�J
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241932

inputs>
,lstm_cell_117_matmul_readvariableop_resource:x@
.lstm_cell_117_matmul_1_readvariableop_resource:x;
-lstm_cell_117_biasadd_readvariableop_resource:x
identity��$lstm_cell_117/BiasAdd/ReadVariableOp�#lstm_cell_117/MatMul/ReadVariableOp�%lstm_cell_117/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_117/MatMul/ReadVariableOpReadVariableOp,lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMulMatMulstrided_slice_2:output:0+lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMul_1MatMulzeros:output:0-lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_117/addAddV2lstm_cell_117/MatMul:product:0 lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_117/BiasAddBiasAddlstm_cell_117/add:z:0,lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_117/splitSplit&lstm_cell_117/split/split_dim:output:0lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_117/SigmoidSigmoidlstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_1Sigmoidlstm_cell_117/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_117/mulMullstm_cell_117/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_117/ReluRelulstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_1Mullstm_cell_117/Sigmoid:y:0 lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_117/add_1AddV2lstm_cell_117/mul:z:0lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_2Sigmoidlstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_117/Relu_1Relulstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_2Mullstm_cell_117/Sigmoid_2:y:0"lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_117_matmul_readvariableop_resource.lstm_cell_117_matmul_1_readvariableop_resource-lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23241848*
condR
while_cond_23241847*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_117/BiasAdd/ReadVariableOp$^lstm_cell_117/MatMul/ReadVariableOp&^lstm_cell_117/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_117/BiasAdd/ReadVariableOp$lstm_cell_117/BiasAdd/ReadVariableOp2J
#lstm_cell_117/MatMul/ReadVariableOp#lstm_cell_117/MatMul/ReadVariableOp2N
%lstm_cell_117/MatMul_1/ReadVariableOp%lstm_cell_117/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23239543

inputs>
,lstm_cell_117_matmul_readvariableop_resource:x@
.lstm_cell_117_matmul_1_readvariableop_resource:x;
-lstm_cell_117_biasadd_readvariableop_resource:x
identity��$lstm_cell_117/BiasAdd/ReadVariableOp�#lstm_cell_117/MatMul/ReadVariableOp�%lstm_cell_117/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_117/MatMul/ReadVariableOpReadVariableOp,lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMulMatMulstrided_slice_2:output:0+lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMul_1MatMulzeros:output:0-lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_117/addAddV2lstm_cell_117/MatMul:product:0 lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_117/BiasAddBiasAddlstm_cell_117/add:z:0,lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_117/splitSplit&lstm_cell_117/split/split_dim:output:0lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_117/SigmoidSigmoidlstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_1Sigmoidlstm_cell_117/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_117/mulMullstm_cell_117/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_117/ReluRelulstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_1Mullstm_cell_117/Sigmoid:y:0 lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_117/add_1AddV2lstm_cell_117/mul:z:0lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_2Sigmoidlstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_117/Relu_1Relulstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_2Mullstm_cell_117/Sigmoid_2:y:0"lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_117_matmul_readvariableop_resource.lstm_cell_117_matmul_1_readvariableop_resource-lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23239459*
condR
while_cond_23239458*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_117/BiasAdd/ReadVariableOp$^lstm_cell_117/MatMul/ReadVariableOp&^lstm_cell_117/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_117/BiasAdd/ReadVariableOp$lstm_cell_117/BiasAdd/ReadVariableOp2J
#lstm_cell_117/MatMul/ReadVariableOp#lstm_cell_117/MatMul/ReadVariableOp2N
%lstm_cell_117/MatMul_1/ReadVariableOp%lstm_cell_117/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

g
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242726

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�S
�
*sequential_89_lstm_112_while_body_23237393J
Fsequential_89_lstm_112_while_sequential_89_lstm_112_while_loop_counterP
Lsequential_89_lstm_112_while_sequential_89_lstm_112_while_maximum_iterations,
(sequential_89_lstm_112_while_placeholder.
*sequential_89_lstm_112_while_placeholder_1.
*sequential_89_lstm_112_while_placeholder_2.
*sequential_89_lstm_112_while_placeholder_3I
Esequential_89_lstm_112_while_sequential_89_lstm_112_strided_slice_1_0�
�sequential_89_lstm_112_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_112_tensorarrayunstack_tensorlistfromtensor_0]
Ksequential_89_lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0:x_
Msequential_89_lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0:xZ
Lsequential_89_lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0:x)
%sequential_89_lstm_112_while_identity+
'sequential_89_lstm_112_while_identity_1+
'sequential_89_lstm_112_while_identity_2+
'sequential_89_lstm_112_while_identity_3+
'sequential_89_lstm_112_while_identity_4+
'sequential_89_lstm_112_while_identity_5G
Csequential_89_lstm_112_while_sequential_89_lstm_112_strided_slice_1�
sequential_89_lstm_112_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_112_tensorarrayunstack_tensorlistfromtensor[
Isequential_89_lstm_112_while_lstm_cell_117_matmul_readvariableop_resource:x]
Ksequential_89_lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource:xX
Jsequential_89_lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource:x��Asequential_89/lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp�@sequential_89/lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp�Bsequential_89/lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp�
Nsequential_89/lstm_112/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
@sequential_89/lstm_112/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_89_lstm_112_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_112_tensorarrayunstack_tensorlistfromtensor_0(sequential_89_lstm_112_while_placeholderWsequential_89/lstm_112/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
@sequential_89/lstm_112/while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOpKsequential_89_lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
1sequential_89/lstm_112/while/lstm_cell_117/MatMulMatMulGsequential_89/lstm_112/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_89/lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
Bsequential_89/lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOpMsequential_89_lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
3sequential_89/lstm_112/while/lstm_cell_117/MatMul_1MatMul*sequential_89_lstm_112_while_placeholder_2Jsequential_89/lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.sequential_89/lstm_112/while/lstm_cell_117/addAddV2;sequential_89/lstm_112/while/lstm_cell_117/MatMul:product:0=sequential_89/lstm_112/while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
Asequential_89/lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOpLsequential_89_lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
2sequential_89/lstm_112/while/lstm_cell_117/BiasAddBiasAdd2sequential_89/lstm_112/while/lstm_cell_117/add:z:0Isequential_89/lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x|
:sequential_89/lstm_112/while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_89/lstm_112/while/lstm_cell_117/splitSplitCsequential_89/lstm_112/while/lstm_cell_117/split/split_dim:output:0;sequential_89/lstm_112/while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
2sequential_89/lstm_112/while/lstm_cell_117/SigmoidSigmoid9sequential_89/lstm_112/while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_112/while/lstm_cell_117/Sigmoid_1Sigmoid9sequential_89/lstm_112/while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
.sequential_89/lstm_112/while/lstm_cell_117/mulMul8sequential_89/lstm_112/while/lstm_cell_117/Sigmoid_1:y:0*sequential_89_lstm_112_while_placeholder_3*
T0*'
_output_shapes
:����������
/sequential_89/lstm_112/while/lstm_cell_117/ReluRelu9sequential_89/lstm_112/while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
0sequential_89/lstm_112/while/lstm_cell_117/mul_1Mul6sequential_89/lstm_112/while/lstm_cell_117/Sigmoid:y:0=sequential_89/lstm_112/while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
0sequential_89/lstm_112/while/lstm_cell_117/add_1AddV22sequential_89/lstm_112/while/lstm_cell_117/mul:z:04sequential_89/lstm_112/while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_112/while/lstm_cell_117/Sigmoid_2Sigmoid9sequential_89/lstm_112/while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:����������
1sequential_89/lstm_112/while/lstm_cell_117/Relu_1Relu4sequential_89/lstm_112/while/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
0sequential_89/lstm_112/while/lstm_cell_117/mul_2Mul8sequential_89/lstm_112/while/lstm_cell_117/Sigmoid_2:y:0?sequential_89/lstm_112/while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
Asequential_89/lstm_112/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_89_lstm_112_while_placeholder_1(sequential_89_lstm_112_while_placeholder4sequential_89/lstm_112/while/lstm_cell_117/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"sequential_89/lstm_112/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_89/lstm_112/while/addAddV2(sequential_89_lstm_112_while_placeholder+sequential_89/lstm_112/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_89/lstm_112/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"sequential_89/lstm_112/while/add_1AddV2Fsequential_89_lstm_112_while_sequential_89_lstm_112_while_loop_counter-sequential_89/lstm_112/while/add_1/y:output:0*
T0*
_output_shapes
: �
%sequential_89/lstm_112/while/IdentityIdentity&sequential_89/lstm_112/while/add_1:z:0"^sequential_89/lstm_112/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_112/while/Identity_1IdentityLsequential_89_lstm_112_while_sequential_89_lstm_112_while_maximum_iterations"^sequential_89/lstm_112/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_112/while/Identity_2Identity$sequential_89/lstm_112/while/add:z:0"^sequential_89/lstm_112/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_112/while/Identity_3IdentityQsequential_89/lstm_112/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_89/lstm_112/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_112/while/Identity_4Identity4sequential_89/lstm_112/while/lstm_cell_117/mul_2:z:0"^sequential_89/lstm_112/while/NoOp*
T0*'
_output_shapes
:����������
'sequential_89/lstm_112/while/Identity_5Identity4sequential_89/lstm_112/while/lstm_cell_117/add_1:z:0"^sequential_89/lstm_112/while/NoOp*
T0*'
_output_shapes
:����������
!sequential_89/lstm_112/while/NoOpNoOpB^sequential_89/lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOpA^sequential_89/lstm_112/while/lstm_cell_117/MatMul/ReadVariableOpC^sequential_89/lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_89_lstm_112_while_identity.sequential_89/lstm_112/while/Identity:output:0"[
'sequential_89_lstm_112_while_identity_10sequential_89/lstm_112/while/Identity_1:output:0"[
'sequential_89_lstm_112_while_identity_20sequential_89/lstm_112/while/Identity_2:output:0"[
'sequential_89_lstm_112_while_identity_30sequential_89/lstm_112/while/Identity_3:output:0"[
'sequential_89_lstm_112_while_identity_40sequential_89/lstm_112/while/Identity_4:output:0"[
'sequential_89_lstm_112_while_identity_50sequential_89/lstm_112/while/Identity_5:output:0"�
Jsequential_89_lstm_112_while_lstm_cell_117_biasadd_readvariableop_resourceLsequential_89_lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0"�
Ksequential_89_lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resourceMsequential_89_lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0"�
Isequential_89_lstm_112_while_lstm_cell_117_matmul_readvariableop_resourceKsequential_89_lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0"�
Csequential_89_lstm_112_while_sequential_89_lstm_112_strided_slice_1Esequential_89_lstm_112_while_sequential_89_lstm_112_strided_slice_1_0"�
sequential_89_lstm_112_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_112_tensorarrayunstack_tensorlistfromtensor�sequential_89_lstm_112_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_112_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2�
Asequential_89/lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOpAsequential_89/lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp2�
@sequential_89/lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp@sequential_89/lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp2�
Bsequential_89/lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOpBsequential_89/lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238392

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�

�
lstm_111_while_cond_23240464.
*lstm_111_while_lstm_111_while_loop_counter4
0lstm_111_while_lstm_111_while_maximum_iterations
lstm_111_while_placeholder 
lstm_111_while_placeholder_1 
lstm_111_while_placeholder_2 
lstm_111_while_placeholder_30
,lstm_111_while_less_lstm_111_strided_slice_1H
Dlstm_111_while_lstm_111_while_cond_23240464___redundant_placeholder0H
Dlstm_111_while_lstm_111_while_cond_23240464___redundant_placeholder1H
Dlstm_111_while_lstm_111_while_cond_23240464___redundant_placeholder2H
Dlstm_111_while_lstm_111_while_cond_23240464___redundant_placeholder3
lstm_111_while_identity
�
lstm_111/while/LessLesslstm_111_while_placeholder,lstm_111_while_less_lstm_111_strided_slice_1*
T0*
_output_shapes
: ]
lstm_111/while/IdentityIdentitylstm_111/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_111_while_identity lstm_111/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
+__inference_lstm_111_layer_call_fn_23240865
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23237966|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�8
�
while_body_23241562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_117_matmul_readvariableop_resource_0:xH
6while_lstm_cell_117_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_117_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_117_matmul_readvariableop_resource:xF
4while_lstm_cell_117_matmul_1_readvariableop_resource:xA
3while_lstm_cell_117_biasadd_readvariableop_resource:x��*while/lstm_cell_117/BiasAdd/ReadVariableOp�)while/lstm_cell_117/MatMul/ReadVariableOp�+while/lstm_cell_117/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_117/addAddV2$while/lstm_cell_117/MatMul:product:0&while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_117/BiasAddBiasAddwhile/lstm_cell_117/add:z:02while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_117/splitSplit,while/lstm_cell_117/split/split_dim:output:0$while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_117/SigmoidSigmoid"while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_1Sigmoid"while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mulMul!while/lstm_cell_117/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_117/ReluRelu"while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_1Mulwhile/lstm_cell_117/Sigmoid:y:0&while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/add_1AddV2while/lstm_cell_117/mul:z:0while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_2Sigmoid"while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_117/Relu_1Reluwhile/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_2Mul!while/lstm_cell_117/Sigmoid_2:y:0(while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_117/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_117/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_117/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_117/BiasAdd/ReadVariableOp*^while/lstm_cell_117/MatMul/ReadVariableOp,^while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_117_biasadd_readvariableop_resource5while_lstm_cell_117_biasadd_readvariableop_resource_0"n
4while_lstm_cell_117_matmul_1_readvariableop_resource6while_lstm_cell_117_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_117_matmul_readvariableop_resource4while_lstm_cell_117_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_117/BiasAdd/ReadVariableOp*while/lstm_cell_117/BiasAdd/ReadVariableOp2V
)while/lstm_cell_117/MatMul/ReadVariableOp)while/lstm_cell_117/MatMul/ReadVariableOp2Z
+while/lstm_cell_117/MatMul_1/ReadVariableOp+while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�D
�

lstm_113_while_body_23240744.
*lstm_113_while_lstm_113_while_loop_counter4
0lstm_113_while_lstm_113_while_maximum_iterations
lstm_113_while_placeholder 
lstm_113_while_placeholder_1 
lstm_113_while_placeholder_2 
lstm_113_while_placeholder_3-
)lstm_113_while_lstm_113_strided_slice_1_0i
elstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0:xQ
?lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0:xL
>lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0:x
lstm_113_while_identity
lstm_113_while_identity_1
lstm_113_while_identity_2
lstm_113_while_identity_3
lstm_113_while_identity_4
lstm_113_while_identity_5+
'lstm_113_while_lstm_113_strided_slice_1g
clstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensorM
;lstm_113_while_lstm_cell_118_matmul_readvariableop_resource:xO
=lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource:xJ
<lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource:x��3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp�2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp�4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp�
@lstm_113/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_113/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensor_0lstm_113_while_placeholderIlstm_113/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp=lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_113/while/lstm_cell_118/MatMulMatMul9lstm_113/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp?lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_113/while/lstm_cell_118/MatMul_1MatMullstm_113_while_placeholder_2<lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_113/while/lstm_cell_118/addAddV2-lstm_113/while/lstm_cell_118/MatMul:product:0/lstm_113/while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp>lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_113/while/lstm_cell_118/BiasAddBiasAdd$lstm_113/while/lstm_cell_118/add:z:0;lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_113/while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_113/while/lstm_cell_118/splitSplit5lstm_113/while/lstm_cell_118/split/split_dim:output:0-lstm_113/while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_113/while/lstm_cell_118/SigmoidSigmoid+lstm_113/while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:����������
&lstm_113/while/lstm_cell_118/Sigmoid_1Sigmoid+lstm_113/while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
 lstm_113/while/lstm_cell_118/mulMul*lstm_113/while/lstm_cell_118/Sigmoid_1:y:0lstm_113_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_113/while/lstm_cell_118/ReluRelu+lstm_113/while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
"lstm_113/while/lstm_cell_118/mul_1Mul(lstm_113/while/lstm_cell_118/Sigmoid:y:0/lstm_113/while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_113/while/lstm_cell_118/add_1AddV2$lstm_113/while/lstm_cell_118/mul:z:0&lstm_113/while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_113/while/lstm_cell_118/Sigmoid_2Sigmoid+lstm_113/while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:����������
#lstm_113/while/lstm_cell_118/Relu_1Relu&lstm_113/while/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_113/while/lstm_cell_118/mul_2Mul*lstm_113/while/lstm_cell_118/Sigmoid_2:y:01lstm_113/while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������{
9lstm_113/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
3lstm_113/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_113_while_placeholder_1Blstm_113/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_113/while/lstm_cell_118/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_113/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_113/while/addAddV2lstm_113_while_placeholderlstm_113/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_113/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_113/while/add_1AddV2*lstm_113_while_lstm_113_while_loop_counterlstm_113/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_113/while/IdentityIdentitylstm_113/while/add_1:z:0^lstm_113/while/NoOp*
T0*
_output_shapes
: �
lstm_113/while/Identity_1Identity0lstm_113_while_lstm_113_while_maximum_iterations^lstm_113/while/NoOp*
T0*
_output_shapes
: t
lstm_113/while/Identity_2Identitylstm_113/while/add:z:0^lstm_113/while/NoOp*
T0*
_output_shapes
: �
lstm_113/while/Identity_3IdentityClstm_113/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_113/while/NoOp*
T0*
_output_shapes
: �
lstm_113/while/Identity_4Identity&lstm_113/while/lstm_cell_118/mul_2:z:0^lstm_113/while/NoOp*
T0*'
_output_shapes
:����������
lstm_113/while/Identity_5Identity&lstm_113/while/lstm_cell_118/add_1:z:0^lstm_113/while/NoOp*
T0*'
_output_shapes
:����������
lstm_113/while/NoOpNoOp4^lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp3^lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp5^lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_113_while_identity lstm_113/while/Identity:output:0"?
lstm_113_while_identity_1"lstm_113/while/Identity_1:output:0"?
lstm_113_while_identity_2"lstm_113/while/Identity_2:output:0"?
lstm_113_while_identity_3"lstm_113/while/Identity_3:output:0"?
lstm_113_while_identity_4"lstm_113/while/Identity_4:output:0"?
lstm_113_while_identity_5"lstm_113/while/Identity_5:output:0"T
'lstm_113_while_lstm_113_strided_slice_1)lstm_113_while_lstm_113_strided_slice_1_0"~
<lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource>lstm_113_while_lstm_cell_118_biasadd_readvariableop_resource_0"�
=lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource?lstm_113_while_lstm_cell_118_matmul_1_readvariableop_resource_0"|
;lstm_113_while_lstm_cell_118_matmul_readvariableop_resource=lstm_113_while_lstm_cell_118_matmul_readvariableop_resource_0"�
clstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensorelstm_113_while_tensorarrayv2read_tensorlistgetitem_lstm_113_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp3lstm_113/while/lstm_cell_118/BiasAdd/ReadVariableOp2h
2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp2lstm_113/while/lstm_cell_118/MatMul/ReadVariableOp2l
4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp4lstm_113/while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�C
�

lstm_111_while_body_23240035.
*lstm_111_while_lstm_111_while_loop_counter4
0lstm_111_while_lstm_111_while_maximum_iterations
lstm_111_while_placeholder 
lstm_111_while_placeholder_1 
lstm_111_while_placeholder_2 
lstm_111_while_placeholder_3-
)lstm_111_while_lstm_111_strided_slice_1_0i
elstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0:xQ
?lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0:xL
>lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0:x
lstm_111_while_identity
lstm_111_while_identity_1
lstm_111_while_identity_2
lstm_111_while_identity_3
lstm_111_while_identity_4
lstm_111_while_identity_5+
'lstm_111_while_lstm_111_strided_slice_1g
clstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensorM
;lstm_111_while_lstm_cell_116_matmul_readvariableop_resource:xO
=lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource:xJ
<lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource:x��3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp�2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp�4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp�
@lstm_111/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_111/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensor_0lstm_111_while_placeholderIlstm_111/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp=lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_111/while/lstm_cell_116/MatMulMatMul9lstm_111/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp?lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_111/while/lstm_cell_116/MatMul_1MatMullstm_111_while_placeholder_2<lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_111/while/lstm_cell_116/addAddV2-lstm_111/while/lstm_cell_116/MatMul:product:0/lstm_111/while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp>lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_111/while/lstm_cell_116/BiasAddBiasAdd$lstm_111/while/lstm_cell_116/add:z:0;lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_111/while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_111/while/lstm_cell_116/splitSplit5lstm_111/while/lstm_cell_116/split/split_dim:output:0-lstm_111/while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_111/while/lstm_cell_116/SigmoidSigmoid+lstm_111/while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:����������
&lstm_111/while/lstm_cell_116/Sigmoid_1Sigmoid+lstm_111/while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
 lstm_111/while/lstm_cell_116/mulMul*lstm_111/while/lstm_cell_116/Sigmoid_1:y:0lstm_111_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_111/while/lstm_cell_116/ReluRelu+lstm_111/while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
"lstm_111/while/lstm_cell_116/mul_1Mul(lstm_111/while/lstm_cell_116/Sigmoid:y:0/lstm_111/while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_111/while/lstm_cell_116/add_1AddV2$lstm_111/while/lstm_cell_116/mul:z:0&lstm_111/while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_111/while/lstm_cell_116/Sigmoid_2Sigmoid+lstm_111/while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:����������
#lstm_111/while/lstm_cell_116/Relu_1Relu&lstm_111/while/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_111/while/lstm_cell_116/mul_2Mul*lstm_111/while/lstm_cell_116/Sigmoid_2:y:01lstm_111/while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_111/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_111_while_placeholder_1lstm_111_while_placeholder&lstm_111/while/lstm_cell_116/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_111/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_111/while/addAddV2lstm_111_while_placeholderlstm_111/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_111/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/while/add_1AddV2*lstm_111_while_lstm_111_while_loop_counterlstm_111/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_111/while/IdentityIdentitylstm_111/while/add_1:z:0^lstm_111/while/NoOp*
T0*
_output_shapes
: �
lstm_111/while/Identity_1Identity0lstm_111_while_lstm_111_while_maximum_iterations^lstm_111/while/NoOp*
T0*
_output_shapes
: t
lstm_111/while/Identity_2Identitylstm_111/while/add:z:0^lstm_111/while/NoOp*
T0*
_output_shapes
: �
lstm_111/while/Identity_3IdentityClstm_111/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_111/while/NoOp*
T0*
_output_shapes
: �
lstm_111/while/Identity_4Identity&lstm_111/while/lstm_cell_116/mul_2:z:0^lstm_111/while/NoOp*
T0*'
_output_shapes
:����������
lstm_111/while/Identity_5Identity&lstm_111/while/lstm_cell_116/add_1:z:0^lstm_111/while/NoOp*
T0*'
_output_shapes
:����������
lstm_111/while/NoOpNoOp4^lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp3^lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp5^lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_111_while_identity lstm_111/while/Identity:output:0"?
lstm_111_while_identity_1"lstm_111/while/Identity_1:output:0"?
lstm_111_while_identity_2"lstm_111/while/Identity_2:output:0"?
lstm_111_while_identity_3"lstm_111/while/Identity_3:output:0"?
lstm_111_while_identity_4"lstm_111/while/Identity_4:output:0"?
lstm_111_while_identity_5"lstm_111/while/Identity_5:output:0"T
'lstm_111_while_lstm_111_strided_slice_1)lstm_111_while_lstm_111_strided_slice_1_0"~
<lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource>lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0"�
=lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource?lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0"|
;lstm_111_while_lstm_cell_116_matmul_readvariableop_resource=lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0"�
clstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensorelstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp2h
2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp2l
4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�9
�
while_body_23242324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_118_matmul_readvariableop_resource_0:xH
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_118_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_118_matmul_readvariableop_resource:xF
4while_lstm_cell_118_matmul_1_readvariableop_resource:xA
3while_lstm_cell_118_biasadd_readvariableop_resource:x��*while/lstm_cell_118/BiasAdd/ReadVariableOp�)while/lstm_cell_118/MatMul/ReadVariableOp�+while/lstm_cell_118/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_118/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_23241089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_116_matmul_readvariableop_resource_0:xH
6while_lstm_cell_116_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_116_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_116_matmul_readvariableop_resource:xF
4while_lstm_cell_116_matmul_1_readvariableop_resource:xA
3while_lstm_cell_116_biasadd_readvariableop_resource:x��*while/lstm_cell_116/BiasAdd/ReadVariableOp�)while/lstm_cell_116/MatMul/ReadVariableOp�+while/lstm_cell_116/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_116/addAddV2$while/lstm_cell_116/MatMul:product:0&while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_116/BiasAddBiasAddwhile/lstm_cell_116/add:z:02while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:0$while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_116/SigmoidSigmoid"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_1Sigmoid"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mulMul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_116/ReluRelu"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_1Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/add_1AddV2while/lstm_cell_116/mul:z:0while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_2Sigmoid"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_2Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_116/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_116/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_116/BiasAdd/ReadVariableOp*^while/lstm_cell_116/MatMul/ReadVariableOp,^while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_116_biasadd_readvariableop_resource5while_lstm_cell_116_biasadd_readvariableop_resource_0"n
4while_lstm_cell_116_matmul_1_readvariableop_resource6while_lstm_cell_116_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_116_matmul_readvariableop_resource4while_lstm_cell_116_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_116/BiasAdd/ReadVariableOp*while/lstm_cell_116/BiasAdd/ReadVariableOp2V
)while/lstm_cell_116/MatMul/ReadVariableOp)while/lstm_cell_116/MatMul/ReadVariableOp2Z
+while/lstm_cell_116/MatMul_1/ReadVariableOp+while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_23241561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23241561___redundant_placeholder06
2while_while_cond_23241561___redundant_placeholder16
2while_while_cond_23241561___redundant_placeholder26
2while_while_cond_23241561___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23238055
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23238055___redundant_placeholder06
2while_while_cond_23238055___redundant_placeholder16
2while_while_cond_23238055___redundant_placeholder26
2while_while_cond_23238055___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�9
�
while_body_23239293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_118_matmul_readvariableop_resource_0:xH
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_118_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_118_matmul_readvariableop_resource:xF
4while_lstm_cell_118_matmul_1_readvariableop_resource:xA
3while_lstm_cell_118_biasadd_readvariableop_resource:x��*while/lstm_cell_118/BiasAdd/ReadVariableOp�)while/lstm_cell_118/MatMul/ReadVariableOp�+while/lstm_cell_118/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_118/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�	
�
F__inference_dense_87_layer_call_and_return_conditional_losses_23239155

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_lstm_cell_117_layer_call_fn_23242860

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238042o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�

�
lstm_113_while_cond_23240743.
*lstm_113_while_lstm_113_while_loop_counter4
0lstm_113_while_lstm_113_while_maximum_iterations
lstm_113_while_placeholder 
lstm_113_while_placeholder_1 
lstm_113_while_placeholder_2 
lstm_113_while_placeholder_30
,lstm_113_while_less_lstm_113_strided_slice_1H
Dlstm_113_while_lstm_113_while_cond_23240743___redundant_placeholder0H
Dlstm_113_while_lstm_113_while_cond_23240743___redundant_placeholder1H
Dlstm_113_while_lstm_113_while_cond_23240743___redundant_placeholder2H
Dlstm_113_while_lstm_113_while_cond_23240743___redundant_placeholder3
lstm_113_while_identity
�
lstm_113/while/LessLesslstm_113_while_placeholder,lstm_113_while_less_lstm_113_strided_slice_1*
T0*
_output_shapes
: ]
lstm_113/while/IdentityIdentitylstm_113/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_113_while_identity lstm_113/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
0__inference_lstm_cell_118_layer_call_fn_23242958

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238392o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�
�
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238540

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�
�
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238042

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�9
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23238670

inputs(
lstm_cell_118_23238586:x(
lstm_cell_118_23238588:x$
lstm_cell_118_23238590:x
identity��%lstm_cell_118/StatefulPartitionedCall�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
%lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_118_23238586lstm_cell_118_23238588lstm_cell_118_23238590*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238540n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_118_23238586lstm_cell_118_23238588lstm_cell_118_23238590*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23238600*
condR
while_cond_23238599*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������v
NoOpNoOp&^lstm_cell_118/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_118/StatefulPartitionedCall%lstm_cell_118/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_23242613
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23242613___redundant_placeholder06
2while_while_cond_23242613___redundant_placeholder16
2while_while_cond_23242613___redundant_placeholder26
2while_while_cond_23242613___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�8
�
while_body_23239624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_116_matmul_readvariableop_resource_0:xH
6while_lstm_cell_116_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_116_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_116_matmul_readvariableop_resource:xF
4while_lstm_cell_116_matmul_1_readvariableop_resource:xA
3while_lstm_cell_116_biasadd_readvariableop_resource:x��*while/lstm_cell_116/BiasAdd/ReadVariableOp�)while/lstm_cell_116/MatMul/ReadVariableOp�+while/lstm_cell_116/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_116/addAddV2$while/lstm_cell_116/MatMul:product:0&while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_116/BiasAddBiasAddwhile/lstm_cell_116/add:z:02while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:0$while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_116/SigmoidSigmoid"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_1Sigmoid"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mulMul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_116/ReluRelu"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_1Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/add_1AddV2while/lstm_cell_116/mul:z:0while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_2Sigmoid"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_2Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_116/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_116/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_116/BiasAdd/ReadVariableOp*^while/lstm_cell_116/MatMul/ReadVariableOp,^while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_116_biasadd_readvariableop_resource5while_lstm_cell_116_biasadd_readvariableop_resource_0"n
4while_lstm_cell_116_matmul_1_readvariableop_resource6while_lstm_cell_116_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_116_matmul_readvariableop_resource4while_lstm_cell_116_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_116/BiasAdd/ReadVariableOp*while/lstm_cell_116/BiasAdd/ReadVariableOp2V
)while/lstm_cell_116/MatMul/ReadVariableOp)while/lstm_cell_116/MatMul/ReadVariableOp2Z
+while/lstm_cell_116/MatMul_1/ReadVariableOp+while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�C
�

lstm_111_while_body_23240465.
*lstm_111_while_lstm_111_while_loop_counter4
0lstm_111_while_lstm_111_while_maximum_iterations
lstm_111_while_placeholder 
lstm_111_while_placeholder_1 
lstm_111_while_placeholder_2 
lstm_111_while_placeholder_3-
)lstm_111_while_lstm_111_strided_slice_1_0i
elstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0:xQ
?lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0:xL
>lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0:x
lstm_111_while_identity
lstm_111_while_identity_1
lstm_111_while_identity_2
lstm_111_while_identity_3
lstm_111_while_identity_4
lstm_111_while_identity_5+
'lstm_111_while_lstm_111_strided_slice_1g
clstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensorM
;lstm_111_while_lstm_cell_116_matmul_readvariableop_resource:xO
=lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource:xJ
<lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource:x��3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp�2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp�4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp�
@lstm_111/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_111/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensor_0lstm_111_while_placeholderIlstm_111/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp=lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_111/while/lstm_cell_116/MatMulMatMul9lstm_111/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp?lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_111/while/lstm_cell_116/MatMul_1MatMullstm_111_while_placeholder_2<lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_111/while/lstm_cell_116/addAddV2-lstm_111/while/lstm_cell_116/MatMul:product:0/lstm_111/while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp>lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_111/while/lstm_cell_116/BiasAddBiasAdd$lstm_111/while/lstm_cell_116/add:z:0;lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_111/while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_111/while/lstm_cell_116/splitSplit5lstm_111/while/lstm_cell_116/split/split_dim:output:0-lstm_111/while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_111/while/lstm_cell_116/SigmoidSigmoid+lstm_111/while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:����������
&lstm_111/while/lstm_cell_116/Sigmoid_1Sigmoid+lstm_111/while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
 lstm_111/while/lstm_cell_116/mulMul*lstm_111/while/lstm_cell_116/Sigmoid_1:y:0lstm_111_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_111/while/lstm_cell_116/ReluRelu+lstm_111/while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
"lstm_111/while/lstm_cell_116/mul_1Mul(lstm_111/while/lstm_cell_116/Sigmoid:y:0/lstm_111/while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_111/while/lstm_cell_116/add_1AddV2$lstm_111/while/lstm_cell_116/mul:z:0&lstm_111/while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_111/while/lstm_cell_116/Sigmoid_2Sigmoid+lstm_111/while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:����������
#lstm_111/while/lstm_cell_116/Relu_1Relu&lstm_111/while/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_111/while/lstm_cell_116/mul_2Mul*lstm_111/while/lstm_cell_116/Sigmoid_2:y:01lstm_111/while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_111/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_111_while_placeholder_1lstm_111_while_placeholder&lstm_111/while/lstm_cell_116/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_111/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_111/while/addAddV2lstm_111_while_placeholderlstm_111/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_111/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_111/while/add_1AddV2*lstm_111_while_lstm_111_while_loop_counterlstm_111/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_111/while/IdentityIdentitylstm_111/while/add_1:z:0^lstm_111/while/NoOp*
T0*
_output_shapes
: �
lstm_111/while/Identity_1Identity0lstm_111_while_lstm_111_while_maximum_iterations^lstm_111/while/NoOp*
T0*
_output_shapes
: t
lstm_111/while/Identity_2Identitylstm_111/while/add:z:0^lstm_111/while/NoOp*
T0*
_output_shapes
: �
lstm_111/while/Identity_3IdentityClstm_111/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_111/while/NoOp*
T0*
_output_shapes
: �
lstm_111/while/Identity_4Identity&lstm_111/while/lstm_cell_116/mul_2:z:0^lstm_111/while/NoOp*
T0*'
_output_shapes
:����������
lstm_111/while/Identity_5Identity&lstm_111/while/lstm_cell_116/add_1:z:0^lstm_111/while/NoOp*
T0*'
_output_shapes
:����������
lstm_111/while/NoOpNoOp4^lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp3^lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp5^lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_111_while_identity lstm_111/while/Identity:output:0"?
lstm_111_while_identity_1"lstm_111/while/Identity_1:output:0"?
lstm_111_while_identity_2"lstm_111/while/Identity_2:output:0"?
lstm_111_while_identity_3"lstm_111/while/Identity_3:output:0"?
lstm_111_while_identity_4"lstm_111/while/Identity_4:output:0"?
lstm_111_while_identity_5"lstm_111/while/Identity_5:output:0"T
'lstm_111_while_lstm_111_strided_slice_1)lstm_111_while_lstm_111_strided_slice_1_0"~
<lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource>lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0"�
=lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource?lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0"|
;lstm_111_while_lstm_cell_116_matmul_readvariableop_resource=lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0"�
clstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensorelstm_111_while_tensorarrayv2read_tensorlistgetitem_lstm_111_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp3lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp2h
2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp2lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp2l
4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp4lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�K
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241030
inputs_0>
,lstm_cell_116_matmul_readvariableop_resource:x@
.lstm_cell_116_matmul_1_readvariableop_resource:x;
-lstm_cell_116_biasadd_readvariableop_resource:x
identity��$lstm_cell_116/BiasAdd/ReadVariableOp�#lstm_cell_116/MatMul/ReadVariableOp�%lstm_cell_116/MatMul_1/ReadVariableOp�while=
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_116/MatMul/ReadVariableOpReadVariableOp,lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0+lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMul_1MatMulzeros:output:0-lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_116/addAddV2lstm_cell_116/MatMul:product:0 lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_116/BiasAddBiasAddlstm_cell_116/add:z:0,lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_116/SigmoidSigmoidlstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_116/mulMullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_116/ReluRelulstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_1Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_116/add_1AddV2lstm_cell_116/mul:z:0lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_116/Relu_1Relulstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_2Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_116_matmul_readvariableop_resource.lstm_cell_116_matmul_1_readvariableop_resource-lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23240946*
condR
while_cond_23240945*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp%^lstm_cell_116/BiasAdd/ReadVariableOp$^lstm_cell_116/MatMul/ReadVariableOp&^lstm_cell_116/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_116/BiasAdd/ReadVariableOp$lstm_cell_116/BiasAdd/ReadVariableOp2J
#lstm_cell_116/MatMul/ReadVariableOp#lstm_cell_116/MatMul/ReadVariableOp2N
%lstm_cell_116/MatMul_1/ReadVariableOp%lstm_cell_116/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237838

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�
�
+__inference_dense_87_layer_call_fn_23242735

inputs
unknown:
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
F__inference_dense_87_layer_call_and_return_conditional_losses_23239155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242941

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�J
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238978

inputs>
,lstm_cell_117_matmul_readvariableop_resource:x@
.lstm_cell_117_matmul_1_readvariableop_resource:x;
-lstm_cell_117_biasadd_readvariableop_resource:x
identity��$lstm_cell_117/BiasAdd/ReadVariableOp�#lstm_cell_117/MatMul/ReadVariableOp�%lstm_cell_117/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_117/MatMul/ReadVariableOpReadVariableOp,lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMulMatMulstrided_slice_2:output:0+lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMul_1MatMulzeros:output:0-lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_117/addAddV2lstm_cell_117/MatMul:product:0 lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_117/BiasAddBiasAddlstm_cell_117/add:z:0,lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_117/splitSplit&lstm_cell_117/split/split_dim:output:0lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_117/SigmoidSigmoidlstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_1Sigmoidlstm_cell_117/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_117/mulMullstm_cell_117/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_117/ReluRelulstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_1Mullstm_cell_117/Sigmoid:y:0 lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_117/add_1AddV2lstm_cell_117/mul:z:0lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_2Sigmoidlstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_117/Relu_1Relulstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_2Mullstm_cell_117/Sigmoid_2:y:0"lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_117_matmul_readvariableop_resource.lstm_cell_117_matmul_1_readvariableop_resource-lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23238894*
condR
while_cond_23238893*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_117/BiasAdd/ReadVariableOp$^lstm_cell_117/MatMul/ReadVariableOp&^lstm_cell_117/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_117/BiasAdd/ReadVariableOp$lstm_cell_117/BiasAdd/ReadVariableOp2J
#lstm_cell_117/MatMul/ReadVariableOp#lstm_cell_117/MatMul/ReadVariableOp2N
%lstm_cell_117/MatMul_1/ReadVariableOp%lstm_cell_117/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

g
H__inference_dropout_70_layer_call_and_return_conditional_losses_23239217

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_23242179
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_118_matmul_readvariableop_resource_0:xH
6while_lstm_cell_118_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_118_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_118_matmul_readvariableop_resource:xF
4while_lstm_cell_118_matmul_1_readvariableop_resource:xA
3while_lstm_cell_118_biasadd_readvariableop_resource:x��*while/lstm_cell_118/BiasAdd/ReadVariableOp�)while/lstm_cell_118/MatMul/ReadVariableOp�+while/lstm_cell_118/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_118/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_118_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_118_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_118/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_118/addAddV2$while/lstm_cell_118/MatMul:product:0&while/lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_118_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_118/BiasAddBiasAddwhile/lstm_cell_118/add:z:02while/lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_118/splitSplit,while/lstm_cell_118/split/split_dim:output:0$while/lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_118/SigmoidSigmoid"while/lstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_1Sigmoid"while/lstm_cell_118/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mulMul!while/lstm_cell_118/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_118/ReluRelu"while/lstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_1Mulwhile/lstm_cell_118/Sigmoid:y:0&while/lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/add_1AddV2while/lstm_cell_118/mul:z:0while/lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_118/Sigmoid_2Sigmoid"while/lstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_118/Relu_1Reluwhile/lstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_118/mul_2Mul!while/lstm_cell_118/Sigmoid_2:y:0(while/lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_118/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_118/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_118/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_118/BiasAdd/ReadVariableOp*^while/lstm_cell_118/MatMul/ReadVariableOp,^while/lstm_cell_118/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_118_biasadd_readvariableop_resource5while_lstm_cell_118_biasadd_readvariableop_resource_0"n
4while_lstm_cell_118_matmul_1_readvariableop_resource6while_lstm_cell_118_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_118_matmul_readvariableop_resource4while_lstm_cell_118_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_118/BiasAdd/ReadVariableOp*while/lstm_cell_118/BiasAdd/ReadVariableOp2V
)while/lstm_cell_118/MatMul/ReadVariableOp)while/lstm_cell_118/MatMul/ReadVariableOp2Z
+while/lstm_cell_118/MatMul_1/ReadVariableOp+while/lstm_cell_118/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�$
�
while_body_23238600
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_118_23238624_0:x0
while_lstm_cell_118_23238626_0:x,
while_lstm_cell_118_23238628_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_118_23238624:x.
while_lstm_cell_118_23238626:x*
while_lstm_cell_118_23238628:x��+while/lstm_cell_118/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_118_23238624_0while_lstm_cell_118_23238626_0while_lstm_cell_118_23238628_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238540r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_118/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_118/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_118/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_118/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_118_23238624while_lstm_cell_118_23238624_0">
while_lstm_cell_118_23238626while_lstm_cell_118_23238626_0">
while_lstm_cell_118_23238628while_lstm_cell_118_23238628_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_118/StatefulPartitionedCall+while/lstm_cell_118/StatefulPartitionedCall: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�8
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238316

inputs(
lstm_cell_117_23238234:x(
lstm_cell_117_23238236:x$
lstm_cell_117_23238238:x
identity��%lstm_cell_117/StatefulPartitionedCall�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
%lstm_cell_117/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_117_23238234lstm_cell_117_23238236lstm_cell_117_23238238*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238188n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_117_23238234lstm_cell_117_23238236lstm_cell_117_23238238*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23238247*
condR
while_cond_23238246*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������v
NoOpNoOp&^lstm_cell_117/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_117/StatefulPartitionedCall%lstm_cell_117/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�8
�
while_body_23238744
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_116_matmul_readvariableop_resource_0:xH
6while_lstm_cell_116_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_116_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_116_matmul_readvariableop_resource:xF
4while_lstm_cell_116_matmul_1_readvariableop_resource:xA
3while_lstm_cell_116_biasadd_readvariableop_resource:x��*while/lstm_cell_116/BiasAdd/ReadVariableOp�)while/lstm_cell_116/MatMul/ReadVariableOp�+while/lstm_cell_116/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_116/addAddV2$while/lstm_cell_116/MatMul:product:0&while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_116/BiasAddBiasAddwhile/lstm_cell_116/add:z:02while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:0$while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_116/SigmoidSigmoid"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_1Sigmoid"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mulMul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_116/ReluRelu"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_1Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/add_1AddV2while/lstm_cell_116/mul:z:0while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_2Sigmoid"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_2Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_116/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_116/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_116/BiasAdd/ReadVariableOp*^while/lstm_cell_116/MatMul/ReadVariableOp,^while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_116_biasadd_readvariableop_resource5while_lstm_cell_116_biasadd_readvariableop_resource_0"n
4while_lstm_cell_116_matmul_1_readvariableop_resource6while_lstm_cell_116_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_116_matmul_readvariableop_resource4while_lstm_cell_116_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_116/BiasAdd/ReadVariableOp*while/lstm_cell_116/BiasAdd/ReadVariableOp2V
)while/lstm_cell_116/MatMul/ReadVariableOp)while/lstm_cell_116/MatMul/ReadVariableOp2Z
+while/lstm_cell_116/MatMul_1/ReadVariableOp+while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_23238893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23238893___redundant_placeholder06
2while_while_cond_23238893___redundant_placeholder16
2while_while_cond_23238893___redundant_placeholder26
2while_while_cond_23238893___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243007

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�
�
while_cond_23242178
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23242178___redundant_placeholder06
2while_while_cond_23242178___redundant_placeholder16
2while_while_cond_23242178___redundant_placeholder26
2while_while_cond_23242178___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23241990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23241990___redundant_placeholder06
2while_while_cond_23241990___redundant_placeholder16
2while_while_cond_23241990___redundant_placeholder26
2while_while_cond_23241990___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�J
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241459

inputs>
,lstm_cell_116_matmul_readvariableop_resource:x@
.lstm_cell_116_matmul_1_readvariableop_resource:x;
-lstm_cell_116_biasadd_readvariableop_resource:x
identity��$lstm_cell_116/BiasAdd/ReadVariableOp�#lstm_cell_116/MatMul/ReadVariableOp�%lstm_cell_116/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_116/MatMul/ReadVariableOpReadVariableOp,lstm_cell_116_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMulMatMulstrided_slice_2:output:0+lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_116_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_116/MatMul_1MatMulzeros:output:0-lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_116/addAddV2lstm_cell_116/MatMul:product:0 lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_116_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_116/BiasAddBiasAddlstm_cell_116/add:z:0,lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_116/splitSplit&lstm_cell_116/split/split_dim:output:0lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_116/SigmoidSigmoidlstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_1Sigmoidlstm_cell_116/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_116/mulMullstm_cell_116/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_116/ReluRelulstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_1Mullstm_cell_116/Sigmoid:y:0 lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_116/add_1AddV2lstm_cell_116/mul:z:0lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_116/Sigmoid_2Sigmoidlstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_116/Relu_1Relulstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_116/mul_2Mullstm_cell_116/Sigmoid_2:y:0"lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_116_matmul_readvariableop_resource.lstm_cell_116_matmul_1_readvariableop_resource-lstm_cell_116_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23241375*
condR
while_cond_23241374*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_116/BiasAdd/ReadVariableOp$^lstm_cell_116/MatMul/ReadVariableOp&^lstm_cell_116/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_116/BiasAdd/ReadVariableOp$lstm_cell_116/BiasAdd/ReadVariableOp2J
#lstm_cell_116/MatMul/ReadVariableOp#lstm_cell_116/MatMul/ReadVariableOp2N
%lstm_cell_116/MatMul_1/ReadVariableOp%lstm_cell_116/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_23238246
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23238246___redundant_placeholder06
2while_while_cond_23238246___redundant_placeholder16
2while_while_cond_23238246___redundant_placeholder26
2while_while_cond_23238246___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�8
�
while_body_23241375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_116_matmul_readvariableop_resource_0:xH
6while_lstm_cell_116_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_116_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_116_matmul_readvariableop_resource:xF
4while_lstm_cell_116_matmul_1_readvariableop_resource:xA
3while_lstm_cell_116_biasadd_readvariableop_resource:x��*while/lstm_cell_116/BiasAdd/ReadVariableOp�)while/lstm_cell_116/MatMul/ReadVariableOp�+while/lstm_cell_116/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_116/addAddV2$while/lstm_cell_116/MatMul:product:0&while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_116/BiasAddBiasAddwhile/lstm_cell_116/add:z:02while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:0$while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_116/SigmoidSigmoid"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_1Sigmoid"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mulMul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_116/ReluRelu"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_1Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/add_1AddV2while/lstm_cell_116/mul:z:0while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_2Sigmoid"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_2Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_116/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_116/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_116/BiasAdd/ReadVariableOp*^while/lstm_cell_116/MatMul/ReadVariableOp,^while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_116_biasadd_readvariableop_resource5while_lstm_cell_116_biasadd_readvariableop_resource_0"n
4while_lstm_cell_116_matmul_1_readvariableop_resource6while_lstm_cell_116_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_116_matmul_readvariableop_resource4while_lstm_cell_116_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_116/BiasAdd/ReadVariableOp*while/lstm_cell_116/BiasAdd/ReadVariableOp2V
)while/lstm_cell_116/MatMul/ReadVariableOp)while/lstm_cell_116/MatMul/ReadVariableOp2Z
+while/lstm_cell_116/MatMul_1/ReadVariableOp+while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�8
�
while_body_23241232
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_116_matmul_readvariableop_resource_0:xH
6while_lstm_cell_116_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_116_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_116_matmul_readvariableop_resource:xF
4while_lstm_cell_116_matmul_1_readvariableop_resource:xA
3while_lstm_cell_116_biasadd_readvariableop_resource:x��*while/lstm_cell_116/BiasAdd/ReadVariableOp�)while/lstm_cell_116/MatMul/ReadVariableOp�+while/lstm_cell_116/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_116/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_116/addAddV2$while/lstm_cell_116/MatMul:product:0&while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_116/BiasAddBiasAddwhile/lstm_cell_116/add:z:02while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_116/splitSplit,while/lstm_cell_116/split/split_dim:output:0$while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_116/SigmoidSigmoid"while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_1Sigmoid"while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mulMul!while/lstm_cell_116/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_116/ReluRelu"while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_1Mulwhile/lstm_cell_116/Sigmoid:y:0&while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/add_1AddV2while/lstm_cell_116/mul:z:0while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_116/Sigmoid_2Sigmoid"while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_116/Relu_1Reluwhile/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_116/mul_2Mul!while/lstm_cell_116/Sigmoid_2:y:0(while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_116/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_116/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_116/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_116/BiasAdd/ReadVariableOp*^while/lstm_cell_116/MatMul/ReadVariableOp,^while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_116_biasadd_readvariableop_resource5while_lstm_cell_116_biasadd_readvariableop_resource_0"n
4while_lstm_cell_116_matmul_1_readvariableop_resource6while_lstm_cell_116_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_116_matmul_readvariableop_resource4while_lstm_cell_116_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_116/BiasAdd/ReadVariableOp*while/lstm_cell_116/BiasAdd/ReadVariableOp2V
)while/lstm_cell_116/MatMul/ReadVariableOp)while/lstm_cell_116/MatMul/ReadVariableOp2Z
+while/lstm_cell_116/MatMul_1/ReadVariableOp+while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�#
�
while_body_23237706
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_116_23237730_0:x0
while_lstm_cell_116_23237732_0:x,
while_lstm_cell_116_23237734_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_116_23237730:x.
while_lstm_cell_116_23237732:x*
while_lstm_cell_116_23237734:x��+while/lstm_cell_116/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_116_23237730_0while_lstm_cell_116_23237732_0while_lstm_cell_116_23237734_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237692�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_116/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_116/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_116/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_116/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_116_23237730while_lstm_cell_116_23237730_0">
while_lstm_cell_116_23237732while_lstm_cell_116_23237732_0">
while_lstm_cell_116_23237734while_lstm_cell_116_23237734_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_116/StatefulPartitionedCall+while/lstm_cell_116/StatefulPartitionedCall: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�

�
lstm_112_while_cond_23240173.
*lstm_112_while_lstm_112_while_loop_counter4
0lstm_112_while_lstm_112_while_maximum_iterations
lstm_112_while_placeholder 
lstm_112_while_placeholder_1 
lstm_112_while_placeholder_2 
lstm_112_while_placeholder_30
,lstm_112_while_less_lstm_112_strided_slice_1H
Dlstm_112_while_lstm_112_while_cond_23240173___redundant_placeholder0H
Dlstm_112_while_lstm_112_while_cond_23240173___redundant_placeholder1H
Dlstm_112_while_lstm_112_while_cond_23240173___redundant_placeholder2H
Dlstm_112_while_lstm_112_while_cond_23240173___redundant_placeholder3
lstm_112_while_identity
�
lstm_112/while/LessLesslstm_112_while_placeholder,lstm_112_while_less_lstm_112_strided_slice_1*
T0*
_output_shapes
: ]
lstm_112/while/IdentityIdentitylstm_112/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_112_while_identity lstm_112/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
+__inference_lstm_113_layer_call_fn_23242097
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23238670o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�8
�
while_body_23241705
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_117_matmul_readvariableop_resource_0:xH
6while_lstm_cell_117_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_117_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_117_matmul_readvariableop_resource:xF
4while_lstm_cell_117_matmul_1_readvariableop_resource:xA
3while_lstm_cell_117_biasadd_readvariableop_resource:x��*while/lstm_cell_117/BiasAdd/ReadVariableOp�)while/lstm_cell_117/MatMul/ReadVariableOp�+while/lstm_cell_117/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_117/addAddV2$while/lstm_cell_117/MatMul:product:0&while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_117/BiasAddBiasAddwhile/lstm_cell_117/add:z:02while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_117/splitSplit,while/lstm_cell_117/split/split_dim:output:0$while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_117/SigmoidSigmoid"while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_1Sigmoid"while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mulMul!while/lstm_cell_117/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_117/ReluRelu"while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_1Mulwhile/lstm_cell_117/Sigmoid:y:0&while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/add_1AddV2while/lstm_cell_117/mul:z:0while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_2Sigmoid"while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_117/Relu_1Reluwhile/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_2Mul!while/lstm_cell_117/Sigmoid_2:y:0(while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_117/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_117/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_117/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_117/BiasAdd/ReadVariableOp*^while/lstm_cell_117/MatMul/ReadVariableOp,^while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_117_biasadd_readvariableop_resource5while_lstm_cell_117_biasadd_readvariableop_resource_0"n
4while_lstm_cell_117_matmul_1_readvariableop_resource6while_lstm_cell_117_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_117_matmul_readvariableop_resource4while_lstm_cell_117_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_117/BiasAdd/ReadVariableOp*while/lstm_cell_117/BiasAdd/ReadVariableOp2V
)while/lstm_cell_117/MatMul/ReadVariableOp)while/lstm_cell_117/MatMul/ReadVariableOp2Z
+while/lstm_cell_117/MatMul_1/ReadVariableOp+while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_lstm_112_layer_call_fn_23241503

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23239543s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23237966

inputs(
lstm_cell_116_23237884:x(
lstm_cell_116_23237886:x$
lstm_cell_116_23237888:x
identity��%lstm_cell_116/StatefulPartitionedCall�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
%lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_116_23237884lstm_cell_116_23237886lstm_cell_116_23237888*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237838n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_116_23237884lstm_cell_116_23237886lstm_cell_116_23237888*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23237897*
condR
while_cond_23237896*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������v
NoOpNoOp&^lstm_cell_116/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_116/StatefulPartitionedCall%lstm_cell_116/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�S
�
*sequential_89_lstm_111_while_body_23237254J
Fsequential_89_lstm_111_while_sequential_89_lstm_111_while_loop_counterP
Lsequential_89_lstm_111_while_sequential_89_lstm_111_while_maximum_iterations,
(sequential_89_lstm_111_while_placeholder.
*sequential_89_lstm_111_while_placeholder_1.
*sequential_89_lstm_111_while_placeholder_2.
*sequential_89_lstm_111_while_placeholder_3I
Esequential_89_lstm_111_while_sequential_89_lstm_111_strided_slice_1_0�
�sequential_89_lstm_111_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_111_tensorarrayunstack_tensorlistfromtensor_0]
Ksequential_89_lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0:x_
Msequential_89_lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0:xZ
Lsequential_89_lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0:x)
%sequential_89_lstm_111_while_identity+
'sequential_89_lstm_111_while_identity_1+
'sequential_89_lstm_111_while_identity_2+
'sequential_89_lstm_111_while_identity_3+
'sequential_89_lstm_111_while_identity_4+
'sequential_89_lstm_111_while_identity_5G
Csequential_89_lstm_111_while_sequential_89_lstm_111_strided_slice_1�
sequential_89_lstm_111_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_111_tensorarrayunstack_tensorlistfromtensor[
Isequential_89_lstm_111_while_lstm_cell_116_matmul_readvariableop_resource:x]
Ksequential_89_lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource:xX
Jsequential_89_lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource:x��Asequential_89/lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp�@sequential_89/lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp�Bsequential_89/lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp�
Nsequential_89/lstm_111/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
@sequential_89/lstm_111/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_89_lstm_111_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_111_tensorarrayunstack_tensorlistfromtensor_0(sequential_89_lstm_111_while_placeholderWsequential_89/lstm_111/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
@sequential_89/lstm_111/while/lstm_cell_116/MatMul/ReadVariableOpReadVariableOpKsequential_89_lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
1sequential_89/lstm_111/while/lstm_cell_116/MatMulMatMulGsequential_89/lstm_111/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_89/lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
Bsequential_89/lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOpReadVariableOpMsequential_89_lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
3sequential_89/lstm_111/while/lstm_cell_116/MatMul_1MatMul*sequential_89_lstm_111_while_placeholder_2Jsequential_89/lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.sequential_89/lstm_111/while/lstm_cell_116/addAddV2;sequential_89/lstm_111/while/lstm_cell_116/MatMul:product:0=sequential_89/lstm_111/while/lstm_cell_116/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
Asequential_89/lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOpReadVariableOpLsequential_89_lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
2sequential_89/lstm_111/while/lstm_cell_116/BiasAddBiasAdd2sequential_89/lstm_111/while/lstm_cell_116/add:z:0Isequential_89/lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x|
:sequential_89/lstm_111/while/lstm_cell_116/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_89/lstm_111/while/lstm_cell_116/splitSplitCsequential_89/lstm_111/while/lstm_cell_116/split/split_dim:output:0;sequential_89/lstm_111/while/lstm_cell_116/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
2sequential_89/lstm_111/while/lstm_cell_116/SigmoidSigmoid9sequential_89/lstm_111/while/lstm_cell_116/split:output:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_111/while/lstm_cell_116/Sigmoid_1Sigmoid9sequential_89/lstm_111/while/lstm_cell_116/split:output:1*
T0*'
_output_shapes
:����������
.sequential_89/lstm_111/while/lstm_cell_116/mulMul8sequential_89/lstm_111/while/lstm_cell_116/Sigmoid_1:y:0*sequential_89_lstm_111_while_placeholder_3*
T0*'
_output_shapes
:����������
/sequential_89/lstm_111/while/lstm_cell_116/ReluRelu9sequential_89/lstm_111/while/lstm_cell_116/split:output:2*
T0*'
_output_shapes
:����������
0sequential_89/lstm_111/while/lstm_cell_116/mul_1Mul6sequential_89/lstm_111/while/lstm_cell_116/Sigmoid:y:0=sequential_89/lstm_111/while/lstm_cell_116/Relu:activations:0*
T0*'
_output_shapes
:����������
0sequential_89/lstm_111/while/lstm_cell_116/add_1AddV22sequential_89/lstm_111/while/lstm_cell_116/mul:z:04sequential_89/lstm_111/while/lstm_cell_116/mul_1:z:0*
T0*'
_output_shapes
:����������
4sequential_89/lstm_111/while/lstm_cell_116/Sigmoid_2Sigmoid9sequential_89/lstm_111/while/lstm_cell_116/split:output:3*
T0*'
_output_shapes
:����������
1sequential_89/lstm_111/while/lstm_cell_116/Relu_1Relu4sequential_89/lstm_111/while/lstm_cell_116/add_1:z:0*
T0*'
_output_shapes
:����������
0sequential_89/lstm_111/while/lstm_cell_116/mul_2Mul8sequential_89/lstm_111/while/lstm_cell_116/Sigmoid_2:y:0?sequential_89/lstm_111/while/lstm_cell_116/Relu_1:activations:0*
T0*'
_output_shapes
:����������
Asequential_89/lstm_111/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_89_lstm_111_while_placeholder_1(sequential_89_lstm_111_while_placeholder4sequential_89/lstm_111/while/lstm_cell_116/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"sequential_89/lstm_111/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_89/lstm_111/while/addAddV2(sequential_89_lstm_111_while_placeholder+sequential_89/lstm_111/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_89/lstm_111/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"sequential_89/lstm_111/while/add_1AddV2Fsequential_89_lstm_111_while_sequential_89_lstm_111_while_loop_counter-sequential_89/lstm_111/while/add_1/y:output:0*
T0*
_output_shapes
: �
%sequential_89/lstm_111/while/IdentityIdentity&sequential_89/lstm_111/while/add_1:z:0"^sequential_89/lstm_111/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_111/while/Identity_1IdentityLsequential_89_lstm_111_while_sequential_89_lstm_111_while_maximum_iterations"^sequential_89/lstm_111/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_111/while/Identity_2Identity$sequential_89/lstm_111/while/add:z:0"^sequential_89/lstm_111/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_111/while/Identity_3IdentityQsequential_89/lstm_111/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_89/lstm_111/while/NoOp*
T0*
_output_shapes
: �
'sequential_89/lstm_111/while/Identity_4Identity4sequential_89/lstm_111/while/lstm_cell_116/mul_2:z:0"^sequential_89/lstm_111/while/NoOp*
T0*'
_output_shapes
:����������
'sequential_89/lstm_111/while/Identity_5Identity4sequential_89/lstm_111/while/lstm_cell_116/add_1:z:0"^sequential_89/lstm_111/while/NoOp*
T0*'
_output_shapes
:����������
!sequential_89/lstm_111/while/NoOpNoOpB^sequential_89/lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOpA^sequential_89/lstm_111/while/lstm_cell_116/MatMul/ReadVariableOpC^sequential_89/lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_89_lstm_111_while_identity.sequential_89/lstm_111/while/Identity:output:0"[
'sequential_89_lstm_111_while_identity_10sequential_89/lstm_111/while/Identity_1:output:0"[
'sequential_89_lstm_111_while_identity_20sequential_89/lstm_111/while/Identity_2:output:0"[
'sequential_89_lstm_111_while_identity_30sequential_89/lstm_111/while/Identity_3:output:0"[
'sequential_89_lstm_111_while_identity_40sequential_89/lstm_111/while/Identity_4:output:0"[
'sequential_89_lstm_111_while_identity_50sequential_89/lstm_111/while/Identity_5:output:0"�
Jsequential_89_lstm_111_while_lstm_cell_116_biasadd_readvariableop_resourceLsequential_89_lstm_111_while_lstm_cell_116_biasadd_readvariableop_resource_0"�
Ksequential_89_lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resourceMsequential_89_lstm_111_while_lstm_cell_116_matmul_1_readvariableop_resource_0"�
Isequential_89_lstm_111_while_lstm_cell_116_matmul_readvariableop_resourceKsequential_89_lstm_111_while_lstm_cell_116_matmul_readvariableop_resource_0"�
Csequential_89_lstm_111_while_sequential_89_lstm_111_strided_slice_1Esequential_89_lstm_111_while_sequential_89_lstm_111_strided_slice_1_0"�
sequential_89_lstm_111_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_111_tensorarrayunstack_tensorlistfromtensor�sequential_89_lstm_111_while_tensorarrayv2read_tensorlistgetitem_sequential_89_lstm_111_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2�
Asequential_89/lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOpAsequential_89/lstm_111/while/lstm_cell_116/BiasAdd/ReadVariableOp2�
@sequential_89/lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp@sequential_89/lstm_111/while/lstm_cell_116/MatMul/ReadVariableOp2�
Bsequential_89/lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOpBsequential_89/lstm_111/while/lstm_cell_116/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
*sequential_89_lstm_112_while_cond_23237392J
Fsequential_89_lstm_112_while_sequential_89_lstm_112_while_loop_counterP
Lsequential_89_lstm_112_while_sequential_89_lstm_112_while_maximum_iterations,
(sequential_89_lstm_112_while_placeholder.
*sequential_89_lstm_112_while_placeholder_1.
*sequential_89_lstm_112_while_placeholder_2.
*sequential_89_lstm_112_while_placeholder_3L
Hsequential_89_lstm_112_while_less_sequential_89_lstm_112_strided_slice_1d
`sequential_89_lstm_112_while_sequential_89_lstm_112_while_cond_23237392___redundant_placeholder0d
`sequential_89_lstm_112_while_sequential_89_lstm_112_while_cond_23237392___redundant_placeholder1d
`sequential_89_lstm_112_while_sequential_89_lstm_112_while_cond_23237392___redundant_placeholder2d
`sequential_89_lstm_112_while_sequential_89_lstm_112_while_cond_23237392___redundant_placeholder3)
%sequential_89_lstm_112_while_identity
�
!sequential_89/lstm_112/while/LessLess(sequential_89_lstm_112_while_placeholderHsequential_89_lstm_112_while_less_sequential_89_lstm_112_strided_slice_1*
T0*
_output_shapes
: y
%sequential_89/lstm_112/while/IdentityIdentity%sequential_89/lstm_112/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_89_lstm_112_while_identity.sequential_89/lstm_112/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23242323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23242323___redundant_placeholder06
2while_while_cond_23242323___redundant_placeholder16
2while_while_cond_23242323___redundant_placeholder26
2while_while_cond_23242323___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�J
�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23242075

inputs>
,lstm_cell_117_matmul_readvariableop_resource:x@
.lstm_cell_117_matmul_1_readvariableop_resource:x;
-lstm_cell_117_biasadd_readvariableop_resource:x
identity��$lstm_cell_117/BiasAdd/ReadVariableOp�#lstm_cell_117/MatMul/ReadVariableOp�%lstm_cell_117/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_117/MatMul/ReadVariableOpReadVariableOp,lstm_cell_117_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMulMatMulstrided_slice_2:output:0+lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_117_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_117/MatMul_1MatMulzeros:output:0-lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_117/addAddV2lstm_cell_117/MatMul:product:0 lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_117_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_117/BiasAddBiasAddlstm_cell_117/add:z:0,lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_117/splitSplit&lstm_cell_117/split/split_dim:output:0lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_117/SigmoidSigmoidlstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_1Sigmoidlstm_cell_117/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_117/mulMullstm_cell_117/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_117/ReluRelulstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_1Mullstm_cell_117/Sigmoid:y:0 lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_117/add_1AddV2lstm_cell_117/mul:z:0lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_117/Sigmoid_2Sigmoidlstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_117/Relu_1Relulstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_117/mul_2Mullstm_cell_117/Sigmoid_2:y:0"lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_117_matmul_readvariableop_resource.lstm_cell_117_matmul_1_readvariableop_resource-lstm_cell_117_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23241991*
condR
while_cond_23241990*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:����������
NoOpNoOp%^lstm_cell_117/BiasAdd/ReadVariableOp$^lstm_cell_117/MatMul/ReadVariableOp&^lstm_cell_117/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_117/BiasAdd/ReadVariableOp$lstm_cell_117/BiasAdd/ReadVariableOp2J
#lstm_cell_117/MatMul/ReadVariableOp#lstm_cell_117/MatMul/ReadVariableOp2N
%lstm_cell_117/MatMul_1/ReadVariableOp%lstm_cell_117/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*sequential_89_lstm_113_while_cond_23237532J
Fsequential_89_lstm_113_while_sequential_89_lstm_113_while_loop_counterP
Lsequential_89_lstm_113_while_sequential_89_lstm_113_while_maximum_iterations,
(sequential_89_lstm_113_while_placeholder.
*sequential_89_lstm_113_while_placeholder_1.
*sequential_89_lstm_113_while_placeholder_2.
*sequential_89_lstm_113_while_placeholder_3L
Hsequential_89_lstm_113_while_less_sequential_89_lstm_113_strided_slice_1d
`sequential_89_lstm_113_while_sequential_89_lstm_113_while_cond_23237532___redundant_placeholder0d
`sequential_89_lstm_113_while_sequential_89_lstm_113_while_cond_23237532___redundant_placeholder1d
`sequential_89_lstm_113_while_sequential_89_lstm_113_while_cond_23237532___redundant_placeholder2d
`sequential_89_lstm_113_while_sequential_89_lstm_113_while_cond_23237532___redundant_placeholder3)
%sequential_89_lstm_113_while_identity
�
!sequential_89/lstm_113/while/LessLess(sequential_89_lstm_113_while_placeholderHsequential_89_lstm_113_while_less_sequential_89_lstm_113_strided_slice_1*
T0*
_output_shapes
: y
%sequential_89/lstm_113/while/IdentityIdentity%sequential_89/lstm_113/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_89_lstm_113_while_identity.sequential_89/lstm_113/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
+__inference_lstm_112_layer_call_fn_23241470
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238125|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242811

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�

�
lstm_112_while_cond_23240603.
*lstm_112_while_lstm_112_while_loop_counter4
0lstm_112_while_lstm_112_while_maximum_iterations
lstm_112_while_placeholder 
lstm_112_while_placeholder_1 
lstm_112_while_placeholder_2 
lstm_112_while_placeholder_30
,lstm_112_while_less_lstm_112_strided_slice_1H
Dlstm_112_while_lstm_112_while_cond_23240603___redundant_placeholder0H
Dlstm_112_while_lstm_112_while_cond_23240603___redundant_placeholder1H
Dlstm_112_while_lstm_112_while_cond_23240603___redundant_placeholder2H
Dlstm_112_while_lstm_112_while_cond_23240603___redundant_placeholder3
lstm_112_while_identity
�
lstm_112/while/LessLesslstm_112_while_placeholder,lstm_112_while_less_lstm_112_strided_slice_1*
T0*
_output_shapes
: ]
lstm_112/while/IdentityIdentitylstm_112/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_112_while_identity lstm_112/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23241847
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23241847___redundant_placeholder06
2while_while_cond_23241847___redundant_placeholder16
2while_while_cond_23241847___redundant_placeholder26
2while_while_cond_23241847___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
+__inference_lstm_112_layer_call_fn_23241481
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238316|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237692

inputs

states
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�

�
lstm_113_while_cond_23240313.
*lstm_113_while_lstm_113_while_loop_counter4
0lstm_113_while_lstm_113_while_maximum_iterations
lstm_113_while_placeholder 
lstm_113_while_placeholder_1 
lstm_113_while_placeholder_2 
lstm_113_while_placeholder_30
,lstm_113_while_less_lstm_113_strided_slice_1H
Dlstm_113_while_lstm_113_while_cond_23240313___redundant_placeholder0H
Dlstm_113_while_lstm_113_while_cond_23240313___redundant_placeholder1H
Dlstm_113_while_lstm_113_while_cond_23240313___redundant_placeholder2H
Dlstm_113_while_lstm_113_while_cond_23240313___redundant_placeholder3
lstm_113_while_identity
�
lstm_113/while/LessLesslstm_113_while_placeholder,lstm_113_while_less_lstm_113_strided_slice_1*
T0*
_output_shapes
: ]
lstm_113/while/IdentityIdentitylstm_113/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_113_while_identity lstm_113/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
*sequential_89_lstm_111_while_cond_23237253J
Fsequential_89_lstm_111_while_sequential_89_lstm_111_while_loop_counterP
Lsequential_89_lstm_111_while_sequential_89_lstm_111_while_maximum_iterations,
(sequential_89_lstm_111_while_placeholder.
*sequential_89_lstm_111_while_placeholder_1.
*sequential_89_lstm_111_while_placeholder_2.
*sequential_89_lstm_111_while_placeholder_3L
Hsequential_89_lstm_111_while_less_sequential_89_lstm_111_strided_slice_1d
`sequential_89_lstm_111_while_sequential_89_lstm_111_while_cond_23237253___redundant_placeholder0d
`sequential_89_lstm_111_while_sequential_89_lstm_111_while_cond_23237253___redundant_placeholder1d
`sequential_89_lstm_111_while_sequential_89_lstm_111_while_cond_23237253___redundant_placeholder2d
`sequential_89_lstm_111_while_sequential_89_lstm_111_while_cond_23237253___redundant_placeholder3)
%sequential_89_lstm_111_while_identity
�
!sequential_89/lstm_111/while/LessLess(sequential_89_lstm_111_while_placeholderHsequential_89_lstm_111_while_less_sequential_89_lstm_111_strided_slice_1*
T0*
_output_shapes
: y
%sequential_89/lstm_111/while/IdentityIdentity%sequential_89/lstm_111/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_89_lstm_111_while_identity.sequential_89/lstm_111/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�C
�

lstm_112_while_body_23240604.
*lstm_112_while_lstm_112_while_loop_counter4
0lstm_112_while_lstm_112_while_maximum_iterations
lstm_112_while_placeholder 
lstm_112_while_placeholder_1 
lstm_112_while_placeholder_2 
lstm_112_while_placeholder_3-
)lstm_112_while_lstm_112_strided_slice_1_0i
elstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0:xQ
?lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0:xL
>lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0:x
lstm_112_while_identity
lstm_112_while_identity_1
lstm_112_while_identity_2
lstm_112_while_identity_3
lstm_112_while_identity_4
lstm_112_while_identity_5+
'lstm_112_while_lstm_112_strided_slice_1g
clstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensorM
;lstm_112_while_lstm_cell_117_matmul_readvariableop_resource:xO
=lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource:xJ
<lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource:x��3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp�2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp�4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp�
@lstm_112/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_112/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensor_0lstm_112_while_placeholderIlstm_112/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp=lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_112/while/lstm_cell_117/MatMulMatMul9lstm_112/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp?lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_112/while/lstm_cell_117/MatMul_1MatMullstm_112_while_placeholder_2<lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_112/while/lstm_cell_117/addAddV2-lstm_112/while/lstm_cell_117/MatMul:product:0/lstm_112/while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp>lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_112/while/lstm_cell_117/BiasAddBiasAdd$lstm_112/while/lstm_cell_117/add:z:0;lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_112/while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_112/while/lstm_cell_117/splitSplit5lstm_112/while/lstm_cell_117/split/split_dim:output:0-lstm_112/while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_112/while/lstm_cell_117/SigmoidSigmoid+lstm_112/while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:����������
&lstm_112/while/lstm_cell_117/Sigmoid_1Sigmoid+lstm_112/while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
 lstm_112/while/lstm_cell_117/mulMul*lstm_112/while/lstm_cell_117/Sigmoid_1:y:0lstm_112_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_112/while/lstm_cell_117/ReluRelu+lstm_112/while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
"lstm_112/while/lstm_cell_117/mul_1Mul(lstm_112/while/lstm_cell_117/Sigmoid:y:0/lstm_112/while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_112/while/lstm_cell_117/add_1AddV2$lstm_112/while/lstm_cell_117/mul:z:0&lstm_112/while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_112/while/lstm_cell_117/Sigmoid_2Sigmoid+lstm_112/while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:����������
#lstm_112/while/lstm_cell_117/Relu_1Relu&lstm_112/while/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_112/while/lstm_cell_117/mul_2Mul*lstm_112/while/lstm_cell_117/Sigmoid_2:y:01lstm_112/while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_112/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_112_while_placeholder_1lstm_112_while_placeholder&lstm_112/while/lstm_cell_117/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_112/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_112/while/addAddV2lstm_112_while_placeholderlstm_112/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_112/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/while/add_1AddV2*lstm_112_while_lstm_112_while_loop_counterlstm_112/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_112/while/IdentityIdentitylstm_112/while/add_1:z:0^lstm_112/while/NoOp*
T0*
_output_shapes
: �
lstm_112/while/Identity_1Identity0lstm_112_while_lstm_112_while_maximum_iterations^lstm_112/while/NoOp*
T0*
_output_shapes
: t
lstm_112/while/Identity_2Identitylstm_112/while/add:z:0^lstm_112/while/NoOp*
T0*
_output_shapes
: �
lstm_112/while/Identity_3IdentityClstm_112/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_112/while/NoOp*
T0*
_output_shapes
: �
lstm_112/while/Identity_4Identity&lstm_112/while/lstm_cell_117/mul_2:z:0^lstm_112/while/NoOp*
T0*'
_output_shapes
:����������
lstm_112/while/Identity_5Identity&lstm_112/while/lstm_cell_117/add_1:z:0^lstm_112/while/NoOp*
T0*'
_output_shapes
:����������
lstm_112/while/NoOpNoOp4^lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp3^lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp5^lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_112_while_identity lstm_112/while/Identity:output:0"?
lstm_112_while_identity_1"lstm_112/while/Identity_1:output:0"?
lstm_112_while_identity_2"lstm_112/while/Identity_2:output:0"?
lstm_112_while_identity_3"lstm_112/while/Identity_3:output:0"?
lstm_112_while_identity_4"lstm_112/while/Identity_4:output:0"?
lstm_112_while_identity_5"lstm_112/while/Identity_5:output:0"T
'lstm_112_while_lstm_112_strided_slice_1)lstm_112_while_lstm_112_strided_slice_1_0"~
<lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource>lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0"�
=lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource?lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0"|
;lstm_112_while_lstm_cell_117_matmul_readvariableop_resource=lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0"�
clstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensorelstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp2h
2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp2l
4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_23242468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23242468___redundant_placeholder06
2while_while_cond_23242468___redundant_placeholder16
2while_while_cond_23242468___redundant_placeholder26
2while_while_cond_23242468___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
0__inference_lstm_cell_117_layer_call_fn_23242877

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238188o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�C
�

lstm_112_while_body_23240174.
*lstm_112_while_lstm_112_while_loop_counter4
0lstm_112_while_lstm_112_while_maximum_iterations
lstm_112_while_placeholder 
lstm_112_while_placeholder_1 
lstm_112_while_placeholder_2 
lstm_112_while_placeholder_3-
)lstm_112_while_lstm_112_strided_slice_1_0i
elstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0:xQ
?lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0:xL
>lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0:x
lstm_112_while_identity
lstm_112_while_identity_1
lstm_112_while_identity_2
lstm_112_while_identity_3
lstm_112_while_identity_4
lstm_112_while_identity_5+
'lstm_112_while_lstm_112_strided_slice_1g
clstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensorM
;lstm_112_while_lstm_cell_117_matmul_readvariableop_resource:xO
=lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource:xJ
<lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource:x��3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp�2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp�4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp�
@lstm_112/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_112/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensor_0lstm_112_while_placeholderIlstm_112/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp=lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_112/while/lstm_cell_117/MatMulMatMul9lstm_112/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp?lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_112/while/lstm_cell_117/MatMul_1MatMullstm_112_while_placeholder_2<lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_112/while/lstm_cell_117/addAddV2-lstm_112/while/lstm_cell_117/MatMul:product:0/lstm_112/while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp>lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_112/while/lstm_cell_117/BiasAddBiasAdd$lstm_112/while/lstm_cell_117/add:z:0;lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_112/while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_112/while/lstm_cell_117/splitSplit5lstm_112/while/lstm_cell_117/split/split_dim:output:0-lstm_112/while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_112/while/lstm_cell_117/SigmoidSigmoid+lstm_112/while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:����������
&lstm_112/while/lstm_cell_117/Sigmoid_1Sigmoid+lstm_112/while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
 lstm_112/while/lstm_cell_117/mulMul*lstm_112/while/lstm_cell_117/Sigmoid_1:y:0lstm_112_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_112/while/lstm_cell_117/ReluRelu+lstm_112/while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
"lstm_112/while/lstm_cell_117/mul_1Mul(lstm_112/while/lstm_cell_117/Sigmoid:y:0/lstm_112/while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_112/while/lstm_cell_117/add_1AddV2$lstm_112/while/lstm_cell_117/mul:z:0&lstm_112/while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_112/while/lstm_cell_117/Sigmoid_2Sigmoid+lstm_112/while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:����������
#lstm_112/while/lstm_cell_117/Relu_1Relu&lstm_112/while/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_112/while/lstm_cell_117/mul_2Mul*lstm_112/while/lstm_cell_117/Sigmoid_2:y:01lstm_112/while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_112/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_112_while_placeholder_1lstm_112_while_placeholder&lstm_112/while/lstm_cell_117/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_112/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_112/while/addAddV2lstm_112_while_placeholderlstm_112/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_112/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_112/while/add_1AddV2*lstm_112_while_lstm_112_while_loop_counterlstm_112/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_112/while/IdentityIdentitylstm_112/while/add_1:z:0^lstm_112/while/NoOp*
T0*
_output_shapes
: �
lstm_112/while/Identity_1Identity0lstm_112_while_lstm_112_while_maximum_iterations^lstm_112/while/NoOp*
T0*
_output_shapes
: t
lstm_112/while/Identity_2Identitylstm_112/while/add:z:0^lstm_112/while/NoOp*
T0*
_output_shapes
: �
lstm_112/while/Identity_3IdentityClstm_112/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_112/while/NoOp*
T0*
_output_shapes
: �
lstm_112/while/Identity_4Identity&lstm_112/while/lstm_cell_117/mul_2:z:0^lstm_112/while/NoOp*
T0*'
_output_shapes
:����������
lstm_112/while/Identity_5Identity&lstm_112/while/lstm_cell_117/add_1:z:0^lstm_112/while/NoOp*
T0*'
_output_shapes
:����������
lstm_112/while/NoOpNoOp4^lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp3^lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp5^lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_112_while_identity lstm_112/while/Identity:output:0"?
lstm_112_while_identity_1"lstm_112/while/Identity_1:output:0"?
lstm_112_while_identity_2"lstm_112/while/Identity_2:output:0"?
lstm_112_while_identity_3"lstm_112/while/Identity_3:output:0"?
lstm_112_while_identity_4"lstm_112/while/Identity_4:output:0"?
lstm_112_while_identity_5"lstm_112/while/Identity_5:output:0"T
'lstm_112_while_lstm_112_strided_slice_1)lstm_112_while_lstm_112_strided_slice_1_0"~
<lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource>lstm_112_while_lstm_cell_117_biasadd_readvariableop_resource_0"�
=lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource?lstm_112_while_lstm_cell_117_matmul_1_readvariableop_resource_0"|
;lstm_112_while_lstm_cell_117_matmul_readvariableop_resource=lstm_112_while_lstm_cell_117_matmul_readvariableop_resource_0"�
clstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensorelstm_112_while_tensorarrayv2read_tensorlistgetitem_lstm_112_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp3lstm_112/while/lstm_cell_117/BiasAdd/ReadVariableOp2h
2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp2lstm_112/while/lstm_cell_117/MatMul/ReadVariableOp2l
4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp4lstm_112/while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
0__inference_lstm_cell_116_layer_call_fn_23242762

inputs
states_0
states_1
unknown:x
	unknown_0:x
	unknown_1:x
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237692o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�
�
while_cond_23239458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23239458___redundant_placeholder06
2while_while_cond_23239458___redundant_placeholder16
2while_while_cond_23239458___redundant_placeholder26
2while_while_cond_23239458___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_23237896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23237896___redundant_placeholder06
2while_while_cond_23237896___redundant_placeholder16
2while_while_cond_23237896___redundant_placeholder26
2while_while_cond_23237896___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�#
�
while_body_23238247
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_117_23238271_0:x0
while_lstm_cell_117_23238273_0:x,
while_lstm_cell_117_23238275_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_117_23238271:x.
while_lstm_cell_117_23238273:x*
while_lstm_cell_117_23238275:x��+while/lstm_cell_117/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_117/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_117_23238271_0while_lstm_cell_117_23238273_0while_lstm_cell_117_23238275_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23238188�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_117/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_117/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_117/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_117/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_117_23238271while_lstm_cell_117_23238271_0">
while_lstm_cell_117_23238273while_lstm_cell_117_23238273_0">
while_lstm_cell_117_23238275while_lstm_cell_117_23238275_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_117/StatefulPartitionedCall+while/lstm_cell_117/StatefulPartitionedCall: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�

�
&__inference_signature_wrapper_23239922
lstm_111_input
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:x
	unknown_7:x
	unknown_8:
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_111_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_23237625o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_111_input
�

�
lstm_111_while_cond_23240034.
*lstm_111_while_lstm_111_while_loop_counter4
0lstm_111_while_lstm_111_while_maximum_iterations
lstm_111_while_placeholder 
lstm_111_while_placeholder_1 
lstm_111_while_placeholder_2 
lstm_111_while_placeholder_30
,lstm_111_while_less_lstm_111_strided_slice_1H
Dlstm_111_while_lstm_111_while_cond_23240034___redundant_placeholder0H
Dlstm_111_while_lstm_111_while_cond_23240034___redundant_placeholder1H
Dlstm_111_while_lstm_111_while_cond_23240034___redundant_placeholder2H
Dlstm_111_while_lstm_111_while_cond_23240034___redundant_placeholder3
lstm_111_while_identity
�
lstm_111/while/LessLesslstm_111_while_placeholder,lstm_111_while_less_lstm_111_strided_slice_1*
T0*
_output_shapes
: ]
lstm_111/while/IdentityIdentitylstm_111/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_111_while_identity lstm_111/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
+__inference_lstm_113_layer_call_fn_23242086
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_113_layer_call_and_return_conditional_losses_23238477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�#
�
while_body_23237897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_116_23237921_0:x0
while_lstm_cell_116_23237923_0:x,
while_lstm_cell_116_23237925_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_116_23237921:x.
while_lstm_cell_116_23237923:x*
while_lstm_cell_116_23237925:x��+while/lstm_cell_116/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_116_23237921_0while_lstm_cell_116_23237923_0while_lstm_cell_116_23237925_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237838�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_116/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_116/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_116/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_116/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_116_23237921while_lstm_cell_116_23237921_0">
while_lstm_cell_116_23237923while_lstm_cell_116_23237923_0">
while_lstm_cell_116_23237925while_lstm_cell_116_23237925_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_116/StatefulPartitionedCall+while/lstm_cell_116/StatefulPartitionedCall: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�

�
0__inference_sequential_89_layer_call_fn_23239949

inputs
unknown:x
	unknown_0:x
	unknown_1:x
	unknown_2:x
	unknown_3:x
	unknown_4:x
	unknown_5:x
	unknown_6:x
	unknown_7:x
	unknown_8:
	unknown_9:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239162o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_23239044
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23239044___redundant_placeholder06
2while_while_cond_23239044___redundant_placeholder16
2while_while_cond_23239044___redundant_placeholder26
2while_while_cond_23239044___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�$
�
while_body_23238407
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_118_23238431_0:x0
while_lstm_cell_118_23238433_0:x,
while_lstm_cell_118_23238435_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_118_23238431:x.
while_lstm_cell_118_23238433:x*
while_lstm_cell_118_23238435:x��+while/lstm_cell_118/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_118/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_118_23238431_0while_lstm_cell_118_23238433_0while_lstm_cell_118_23238435_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23238392r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_118/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_118/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_118/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_118/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_118_23238431while_lstm_cell_118_23238431_0">
while_lstm_cell_118_23238433while_lstm_cell_118_23238433_0">
while_lstm_cell_118_23238435while_lstm_cell_118_23238435_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_118/StatefulPartitionedCall+while/lstm_cell_118/StatefulPartitionedCall: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�K
�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242699

inputs>
,lstm_cell_118_matmul_readvariableop_resource:x@
.lstm_cell_118_matmul_1_readvariableop_resource:x;
-lstm_cell_118_biasadd_readvariableop_resource:x
identity��$lstm_cell_118/BiasAdd/ReadVariableOp�#lstm_cell_118/MatMul/ReadVariableOp�%lstm_cell_118/MatMul_1/ReadVariableOp�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
#lstm_cell_118/MatMul/ReadVariableOpReadVariableOp,lstm_cell_118_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMulMatMulstrided_slice_2:output:0+lstm_cell_118/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_118/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_118_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_118/MatMul_1MatMulzeros:output:0-lstm_cell_118/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_118/addAddV2lstm_cell_118/MatMul:product:0 lstm_cell_118/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_118/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_118_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_118/BiasAddBiasAddlstm_cell_118/add:z:0,lstm_cell_118/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_118/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_118/splitSplit&lstm_cell_118/split/split_dim:output:0lstm_cell_118/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_118/SigmoidSigmoidlstm_cell_118/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_1Sigmoidlstm_cell_118/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_118/mulMullstm_cell_118/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_118/ReluRelulstm_cell_118/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_1Mullstm_cell_118/Sigmoid:y:0 lstm_cell_118/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_118/add_1AddV2lstm_cell_118/mul:z:0lstm_cell_118/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_118/Sigmoid_2Sigmoidlstm_cell_118/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_118/Relu_1Relulstm_cell_118/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_118/mul_2Mullstm_cell_118/Sigmoid_2:y:0"lstm_cell_118/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   ^
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_118_matmul_readvariableop_resource.lstm_cell_118_matmul_1_readvariableop_resource-lstm_cell_118_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23242614*
condR
while_cond_23242613*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^lstm_cell_118/BiasAdd/ReadVariableOp$^lstm_cell_118/MatMul/ReadVariableOp&^lstm_cell_118/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_118/BiasAdd/ReadVariableOp$lstm_cell_118/BiasAdd/ReadVariableOp2J
#lstm_cell_118/MatMul/ReadVariableOp#lstm_cell_118/MatMul/ReadVariableOp2N
%lstm_cell_118/MatMul_1/ReadVariableOp%lstm_cell_118/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242843

inputs
states_0
states_10
matmul_readvariableop_resource:x2
 matmul_1_readvariableop_resource:x-
biasadd_readvariableop_resource:x
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1
�8
�
while_body_23238894
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_117_matmul_readvariableop_resource_0:xH
6while_lstm_cell_117_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_117_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_117_matmul_readvariableop_resource:xF
4while_lstm_cell_117_matmul_1_readvariableop_resource:xA
3while_lstm_cell_117_biasadd_readvariableop_resource:x��*while/lstm_cell_117/BiasAdd/ReadVariableOp�)while/lstm_cell_117/MatMul/ReadVariableOp�+while/lstm_cell_117/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_117/addAddV2$while/lstm_cell_117/MatMul:product:0&while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_117/BiasAddBiasAddwhile/lstm_cell_117/add:z:02while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_117/splitSplit,while/lstm_cell_117/split/split_dim:output:0$while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_117/SigmoidSigmoid"while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_1Sigmoid"while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mulMul!while/lstm_cell_117/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_117/ReluRelu"while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_1Mulwhile/lstm_cell_117/Sigmoid:y:0&while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/add_1AddV2while/lstm_cell_117/mul:z:0while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_2Sigmoid"while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_117/Relu_1Reluwhile/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_2Mul!while/lstm_cell_117/Sigmoid_2:y:0(while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_117/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_117/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_117/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_117/BiasAdd/ReadVariableOp*^while/lstm_cell_117/MatMul/ReadVariableOp,^while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_117_biasadd_readvariableop_resource5while_lstm_cell_117_biasadd_readvariableop_resource_0"n
4while_lstm_cell_117_matmul_1_readvariableop_resource6while_lstm_cell_117_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_117_matmul_readvariableop_resource4while_lstm_cell_117_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_117/BiasAdd/ReadVariableOp*while/lstm_cell_117/BiasAdd/ReadVariableOp2V
)while/lstm_cell_117/MatMul/ReadVariableOp)while/lstm_cell_117/MatMul/ReadVariableOp2Z
+while/lstm_cell_117/MatMul_1/ReadVariableOp+while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�8
�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23237775

inputs(
lstm_cell_116_23237693:x(
lstm_cell_116_23237695:x$
lstm_cell_116_23237697:x
identity��%lstm_cell_116/StatefulPartitionedCall�while;
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
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
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
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
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
valueB"����   �
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
:���������*
shrink_axis_mask�
%lstm_cell_116/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_116_23237693lstm_cell_116_23237695lstm_cell_116_23237697*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23237692n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_116_23237693lstm_cell_116_23237695lstm_cell_116_23237697*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23237706*
condR
while_cond_23237705*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
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
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������v
NoOpNoOp&^lstm_cell_116/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_116/StatefulPartitionedCall%lstm_cell_116/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
+__inference_lstm_112_layer_call_fn_23241492

inputs
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_112_layer_call_and_return_conditional_losses_23238978s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_23241848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_117_matmul_readvariableop_resource_0:xH
6while_lstm_cell_117_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_117_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_117_matmul_readvariableop_resource:xF
4while_lstm_cell_117_matmul_1_readvariableop_resource:xA
3while_lstm_cell_117_biasadd_readvariableop_resource:x��*while/lstm_cell_117/BiasAdd/ReadVariableOp�)while/lstm_cell_117/MatMul/ReadVariableOp�+while/lstm_cell_117/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_117/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_117_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_117/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_117/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_117_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_117/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_117/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_117/addAddV2$while/lstm_cell_117/MatMul:product:0&while/lstm_cell_117/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_117/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_117_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_117/BiasAddBiasAddwhile/lstm_cell_117/add:z:02while/lstm_cell_117/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_117/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_117/splitSplit,while/lstm_cell_117/split/split_dim:output:0$while/lstm_cell_117/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_117/SigmoidSigmoid"while/lstm_cell_117/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_1Sigmoid"while/lstm_cell_117/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mulMul!while/lstm_cell_117/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_117/ReluRelu"while/lstm_cell_117/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_1Mulwhile/lstm_cell_117/Sigmoid:y:0&while/lstm_cell_117/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/add_1AddV2while/lstm_cell_117/mul:z:0while/lstm_cell_117/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_117/Sigmoid_2Sigmoid"while/lstm_cell_117/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_117/Relu_1Reluwhile/lstm_cell_117/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_117/mul_2Mul!while/lstm_cell_117/Sigmoid_2:y:0(while/lstm_cell_117/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_117/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_117/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_117/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_117/BiasAdd/ReadVariableOp*^while/lstm_cell_117/MatMul/ReadVariableOp,^while/lstm_cell_117/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_117_biasadd_readvariableop_resource5while_lstm_cell_117_biasadd_readvariableop_resource_0"n
4while_lstm_cell_117_matmul_1_readvariableop_resource6while_lstm_cell_117_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_117_matmul_readvariableop_resource4while_lstm_cell_117_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_117/BiasAdd/ReadVariableOp*while/lstm_cell_117/BiasAdd/ReadVariableOp2V
)while/lstm_cell_117/MatMul/ReadVariableOp)while/lstm_cell_117/MatMul/ReadVariableOp2Z
+while/lstm_cell_117/MatMul_1/ReadVariableOp+while/lstm_cell_117/MatMul_1/ReadVariableOp: 
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
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_lstm_111_layer_call_fn_23240854
inputs_0
unknown:x
	unknown_0:x
	unknown_1:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_111_layer_call_and_return_conditional_losses_23237775|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
while_cond_23239292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23239292___redundant_placeholder06
2while_while_cond_23239292___redundant_placeholder16
2while_while_cond_23239292___redundant_placeholder26
2while_while_cond_23239292___redundant_placeholder3
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
@: : : : :���������:���������: ::::: 
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
:���������:-)
'
_output_shapes
:���������:
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
M
lstm_111_input;
 serving_default_lstm_111_input:0���������<
dense_870
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell
 
state_spec"
_tf_keras_rnn_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_random_generator
(cell
)
state_spec"
_tf_keras_rnn_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
n
90
:1
;2
<3
=4
>5
?6
@7
A8
79
810"
trackable_list_wrapper
n
90
:1
;2
<3
=4
>5
?6
@7
A8
79
810"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32�
0__inference_sequential_89_layer_call_fn_23239187
0__inference_sequential_89_layer_call_fn_23239949
0__inference_sequential_89_layer_call_fn_23239976
0__inference_sequential_89_layer_call_fn_23239829�
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
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
�
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32�
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240406
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240843
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239860
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239891�
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
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
�B�
#__inference__wrapped_model_23237625lstm_111_input"�
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
O
_variables
P_iterations
Q_learning_rate
R_index_dict
S
_momentums
T_velocities
U_update_step_xla"
experimentalOptimizer
,
Vserving_default"
signature_map
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Wstates
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
]trace_0
^trace_1
_trace_2
`trace_32�
+__inference_lstm_111_layer_call_fn_23240854
+__inference_lstm_111_layer_call_fn_23240865
+__inference_lstm_111_layer_call_fn_23240876
+__inference_lstm_111_layer_call_fn_23240887�
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
 z]trace_0z^trace_1z_trace_2z`trace_3
�
atrace_0
btrace_1
ctrace_2
dtrace_32�
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241030
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241173
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241316
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241459�
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
 zatrace_0zbtrace_1zctrace_2zdtrace_3
"
_generic_user_object
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_random_generator
l
state_size

9kernel
:recurrent_kernel
;bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

mstates
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
strace_0
ttrace_1
utrace_2
vtrace_32�
+__inference_lstm_112_layer_call_fn_23241470
+__inference_lstm_112_layer_call_fn_23241481
+__inference_lstm_112_layer_call_fn_23241492
+__inference_lstm_112_layer_call_fn_23241503�
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
 zstrace_0zttrace_1zutrace_2zvtrace_3
�
wtrace_0
xtrace_1
ytrace_2
ztrace_32�
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241646
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241789
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241932
F__inference_lstm_112_layer_call_and_return_conditional_losses_23242075�
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
 zwtrace_0zxtrace_1zytrace_2zztrace_3
"
_generic_user_object
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size

<kernel
=recurrent_kernel
>bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
+__inference_lstm_113_layer_call_fn_23242086
+__inference_lstm_113_layer_call_fn_23242097
+__inference_lstm_113_layer_call_fn_23242108
+__inference_lstm_113_layer_call_fn_23242119�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242264
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242409
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242554
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242699�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size

?kernel
@recurrent_kernel
Abias"
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_dropout_70_layer_call_fn_23242704
-__inference_dropout_70_layer_call_fn_23242709�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242714
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242726�
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
 z�trace_0z�trace_1
"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_87_layer_call_fn_23242735�
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
 z�trace_0
�
�trace_02�
F__inference_dense_87_layer_call_and_return_conditional_losses_23242745�
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
 z�trace_0
!:2dense_87/kernel
:2dense_87/bias
/:-x2lstm_111/lstm_cell_116/kernel
9:7x2'lstm_111/lstm_cell_116/recurrent_kernel
):'x2lstm_111/lstm_cell_116/bias
/:-x2lstm_112/lstm_cell_117/kernel
9:7x2'lstm_112/lstm_cell_117/recurrent_kernel
):'x2lstm_112/lstm_cell_117/bias
/:-x2lstm_113/lstm_cell_118/kernel
9:7x2'lstm_113/lstm_cell_118/recurrent_kernel
):'x2lstm_113/lstm_cell_118/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_sequential_89_layer_call_fn_23239187lstm_111_input"�
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
0__inference_sequential_89_layer_call_fn_23239949inputs"�
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
0__inference_sequential_89_layer_call_fn_23239976inputs"�
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
0__inference_sequential_89_layer_call_fn_23239829lstm_111_input"�
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
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240406inputs"�
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
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240843inputs"�
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
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239860lstm_111_input"�
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
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239891lstm_111_input"�
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
�
P0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
y
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10"
trackable_list_wrapper
y
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10"
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
&__inference_signature_wrapper_23239922lstm_111_input"�
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
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_lstm_111_layer_call_fn_23240854inputs_0"�
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
+__inference_lstm_111_layer_call_fn_23240865inputs_0"�
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
+__inference_lstm_111_layer_call_fn_23240876inputs"�
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
+__inference_lstm_111_layer_call_fn_23240887inputs"�
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
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241030inputs_0"�
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
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241173inputs_0"�
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
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241316inputs"�
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
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241459inputs"�
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
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_lstm_cell_116_layer_call_fn_23242762
0__inference_lstm_cell_116_layer_call_fn_23242779�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242811
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242843�
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
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_lstm_112_layer_call_fn_23241470inputs_0"�
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
+__inference_lstm_112_layer_call_fn_23241481inputs_0"�
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
+__inference_lstm_112_layer_call_fn_23241492inputs"�
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
+__inference_lstm_112_layer_call_fn_23241503inputs"�
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
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241646inputs_0"�
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
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241789inputs_0"�
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
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241932inputs"�
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
F__inference_lstm_112_layer_call_and_return_conditional_losses_23242075inputs"�
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
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_lstm_cell_117_layer_call_fn_23242860
0__inference_lstm_cell_117_layer_call_fn_23242877�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242909
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242941�
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
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_lstm_113_layer_call_fn_23242086inputs_0"�
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
+__inference_lstm_113_layer_call_fn_23242097inputs_0"�
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
+__inference_lstm_113_layer_call_fn_23242108inputs"�
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
+__inference_lstm_113_layer_call_fn_23242119inputs"�
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
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242264inputs_0"�
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
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242409inputs_0"�
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
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242554inputs"�
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
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242699inputs"�
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
?0
@1
A2"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_lstm_cell_118_layer_call_fn_23242958
0__inference_lstm_cell_118_layer_call_fn_23242975�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243007
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243039�
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
 z�trace_0z�trace_1
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
-__inference_dropout_70_layer_call_fn_23242704inputs"�
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
-__inference_dropout_70_layer_call_fn_23242709inputs"�
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
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242714inputs"�
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
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242726inputs"�
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
+__inference_dense_87_layer_call_fn_23242735inputs"�
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
F__inference_dense_87_layer_call_and_return_conditional_losses_23242745inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
4:2x2$Adam/m/lstm_111/lstm_cell_116/kernel
4:2x2$Adam/v/lstm_111/lstm_cell_116/kernel
>:<x2.Adam/m/lstm_111/lstm_cell_116/recurrent_kernel
>:<x2.Adam/v/lstm_111/lstm_cell_116/recurrent_kernel
.:,x2"Adam/m/lstm_111/lstm_cell_116/bias
.:,x2"Adam/v/lstm_111/lstm_cell_116/bias
4:2x2$Adam/m/lstm_112/lstm_cell_117/kernel
4:2x2$Adam/v/lstm_112/lstm_cell_117/kernel
>:<x2.Adam/m/lstm_112/lstm_cell_117/recurrent_kernel
>:<x2.Adam/v/lstm_112/lstm_cell_117/recurrent_kernel
.:,x2"Adam/m/lstm_112/lstm_cell_117/bias
.:,x2"Adam/v/lstm_112/lstm_cell_117/bias
4:2x2$Adam/m/lstm_113/lstm_cell_118/kernel
4:2x2$Adam/v/lstm_113/lstm_cell_118/kernel
>:<x2.Adam/m/lstm_113/lstm_cell_118/recurrent_kernel
>:<x2.Adam/v/lstm_113/lstm_cell_118/recurrent_kernel
.:,x2"Adam/m/lstm_113/lstm_cell_118/bias
.:,x2"Adam/v/lstm_113/lstm_cell_118/bias
&:$2Adam/m/dense_87/kernel
&:$2Adam/v/dense_87/kernel
 :2Adam/m/dense_87/bias
 :2Adam/v/dense_87/bias
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
0__inference_lstm_cell_116_layer_call_fn_23242762inputsstates_0states_1"�
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
0__inference_lstm_cell_116_layer_call_fn_23242779inputsstates_0states_1"�
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
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242811inputsstates_0states_1"�
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
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242843inputsstates_0states_1"�
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
0__inference_lstm_cell_117_layer_call_fn_23242860inputsstates_0states_1"�
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
0__inference_lstm_cell_117_layer_call_fn_23242877inputsstates_0states_1"�
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
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242909inputsstates_0states_1"�
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
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242941inputsstates_0states_1"�
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
0__inference_lstm_cell_118_layer_call_fn_23242958inputsstates_0states_1"�
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
0__inference_lstm_cell_118_layer_call_fn_23242975inputsstates_0states_1"�
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
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243007inputsstates_0states_1"�
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
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243039inputsstates_0states_1"�
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
#__inference__wrapped_model_232376259:;<=>?@A78;�8
1�.
,�)
lstm_111_input���������
� "3�0
.
dense_87"�
dense_87����������
F__inference_dense_87_layer_call_and_return_conditional_losses_23242745c78/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_87_layer_call_fn_23242735X78/�,
%�"
 �
inputs���������
� "!�
unknown����������
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242714c3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
H__inference_dropout_70_layer_call_and_return_conditional_losses_23242726c3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
-__inference_dropout_70_layer_call_fn_23242704X3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
-__inference_dropout_70_layer_call_fn_23242709X3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241030�9:;O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "9�6
/�,
tensor_0������������������
� �
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241173�9:;O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "9�6
/�,
tensor_0������������������
� �
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241316x9:;?�<
5�2
$�!
inputs���������

 
p 

 
� "0�-
&�#
tensor_0���������
� �
F__inference_lstm_111_layer_call_and_return_conditional_losses_23241459x9:;?�<
5�2
$�!
inputs���������

 
p

 
� "0�-
&�#
tensor_0���������
� �
+__inference_lstm_111_layer_call_fn_23240854�9:;O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ".�+
unknown�������������������
+__inference_lstm_111_layer_call_fn_23240865�9:;O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ".�+
unknown�������������������
+__inference_lstm_111_layer_call_fn_23240876m9:;?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
unknown����������
+__inference_lstm_111_layer_call_fn_23240887m9:;?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
unknown����������
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241646�<=>O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "9�6
/�,
tensor_0������������������
� �
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241789�<=>O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "9�6
/�,
tensor_0������������������
� �
F__inference_lstm_112_layer_call_and_return_conditional_losses_23241932x<=>?�<
5�2
$�!
inputs���������

 
p 

 
� "0�-
&�#
tensor_0���������
� �
F__inference_lstm_112_layer_call_and_return_conditional_losses_23242075x<=>?�<
5�2
$�!
inputs���������

 
p

 
� "0�-
&�#
tensor_0���������
� �
+__inference_lstm_112_layer_call_fn_23241470�<=>O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ".�+
unknown�������������������
+__inference_lstm_112_layer_call_fn_23241481�<=>O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ".�+
unknown�������������������
+__inference_lstm_112_layer_call_fn_23241492m<=>?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
unknown����������
+__inference_lstm_112_layer_call_fn_23241503m<=>?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
unknown����������
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242264�?@AO�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ",�)
"�
tensor_0���������
� �
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242409�?@AO�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ",�)
"�
tensor_0���������
� �
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242554t?@A?�<
5�2
$�!
inputs���������

 
p 

 
� ",�)
"�
tensor_0���������
� �
F__inference_lstm_113_layer_call_and_return_conditional_losses_23242699t?@A?�<
5�2
$�!
inputs���������

 
p

 
� ",�)
"�
tensor_0���������
� �
+__inference_lstm_113_layer_call_fn_23242086y?@AO�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "!�
unknown����������
+__inference_lstm_113_layer_call_fn_23242097y?@AO�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "!�
unknown����������
+__inference_lstm_113_layer_call_fn_23242108i?@A?�<
5�2
$�!
inputs���������

 
p 

 
� "!�
unknown����������
+__inference_lstm_113_layer_call_fn_23242119i?@A?�<
5�2
$�!
inputs���������

 
p

 
� "!�
unknown����������
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242811�9:;��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
K__inference_lstm_cell_116_layer_call_and_return_conditional_losses_23242843�9:;��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
0__inference_lstm_cell_116_layer_call_fn_23242762�9:;��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
0__inference_lstm_cell_116_layer_call_fn_23242779�9:;��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242909�<=>��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
K__inference_lstm_cell_117_layer_call_and_return_conditional_losses_23242941�<=>��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
0__inference_lstm_cell_117_layer_call_fn_23242860�<=>��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
0__inference_lstm_cell_117_layer_call_fn_23242877�<=>��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243007�?@A��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
K__inference_lstm_cell_118_layer_call_and_return_conditional_losses_23243039�?@A��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
0__inference_lstm_cell_118_layer_call_fn_23242958�?@A��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
0__inference_lstm_cell_118_layer_call_fn_23242975�?@A��}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239860�9:;<=>?@A78C�@
9�6
,�)
lstm_111_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_89_layer_call_and_return_conditional_losses_23239891�9:;<=>?@A78C�@
9�6
,�)
lstm_111_input���������
p

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240406x9:;<=>?@A78;�8
1�.
$�!
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_89_layer_call_and_return_conditional_losses_23240843x9:;<=>?@A78;�8
1�.
$�!
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
0__inference_sequential_89_layer_call_fn_23239187u9:;<=>?@A78C�@
9�6
,�)
lstm_111_input���������
p 

 
� "!�
unknown����������
0__inference_sequential_89_layer_call_fn_23239829u9:;<=>?@A78C�@
9�6
,�)
lstm_111_input���������
p

 
� "!�
unknown����������
0__inference_sequential_89_layer_call_fn_23239949m9:;<=>?@A78;�8
1�.
$�!
inputs���������
p 

 
� "!�
unknown����������
0__inference_sequential_89_layer_call_fn_23239976m9:;<=>?@A78;�8
1�.
$�!
inputs���������
p

 
� "!�
unknown����������
&__inference_signature_wrapper_23239922�9:;<=>?@A78M�J
� 
C�@
>
lstm_111_input,�)
lstm_111_input���������"3�0
.
dense_87"�
dense_87���������