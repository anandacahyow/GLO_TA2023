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
Adam/v/dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_85/bias
y
(Adam/v/dense_85/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_85/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_85/bias
y
(Adam/m/dense_85/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_85/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_85/kernel
�
*Adam/v/dense_85/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_85/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_85/kernel
�
*Adam/m/dense_85/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_85/kernel*
_output_shapes

:*
dtype0
�
"Adam/v/lstm_107/lstm_cell_112/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/v/lstm_107/lstm_cell_112/bias
�
6Adam/v/lstm_107/lstm_cell_112/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_107/lstm_cell_112/bias*
_output_shapes
:x*
dtype0
�
"Adam/m/lstm_107/lstm_cell_112/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/m/lstm_107/lstm_cell_112/bias
�
6Adam/m/lstm_107/lstm_cell_112/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_107/lstm_cell_112/bias*
_output_shapes
:x*
dtype0
�
.Adam/v/lstm_107/lstm_cell_112/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/v/lstm_107/lstm_cell_112/recurrent_kernel
�
BAdam/v/lstm_107/lstm_cell_112/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_107/lstm_cell_112/recurrent_kernel*
_output_shapes

:x*
dtype0
�
.Adam/m/lstm_107/lstm_cell_112/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/m/lstm_107/lstm_cell_112/recurrent_kernel
�
BAdam/m/lstm_107/lstm_cell_112/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_107/lstm_cell_112/recurrent_kernel*
_output_shapes

:x*
dtype0
�
$Adam/v/lstm_107/lstm_cell_112/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/v/lstm_107/lstm_cell_112/kernel
�
8Adam/v/lstm_107/lstm_cell_112/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_107/lstm_cell_112/kernel*
_output_shapes

:x*
dtype0
�
$Adam/m/lstm_107/lstm_cell_112/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/m/lstm_107/lstm_cell_112/kernel
�
8Adam/m/lstm_107/lstm_cell_112/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_107/lstm_cell_112/kernel*
_output_shapes

:x*
dtype0
�
"Adam/v/lstm_106/lstm_cell_111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/v/lstm_106/lstm_cell_111/bias
�
6Adam/v/lstm_106/lstm_cell_111/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_106/lstm_cell_111/bias*
_output_shapes
:x*
dtype0
�
"Adam/m/lstm_106/lstm_cell_111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/m/lstm_106/lstm_cell_111/bias
�
6Adam/m/lstm_106/lstm_cell_111/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_106/lstm_cell_111/bias*
_output_shapes
:x*
dtype0
�
.Adam/v/lstm_106/lstm_cell_111/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/v/lstm_106/lstm_cell_111/recurrent_kernel
�
BAdam/v/lstm_106/lstm_cell_111/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_106/lstm_cell_111/recurrent_kernel*
_output_shapes

:x*
dtype0
�
.Adam/m/lstm_106/lstm_cell_111/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/m/lstm_106/lstm_cell_111/recurrent_kernel
�
BAdam/m/lstm_106/lstm_cell_111/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_106/lstm_cell_111/recurrent_kernel*
_output_shapes

:x*
dtype0
�
$Adam/v/lstm_106/lstm_cell_111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/v/lstm_106/lstm_cell_111/kernel
�
8Adam/v/lstm_106/lstm_cell_111/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_106/lstm_cell_111/kernel*
_output_shapes

:x*
dtype0
�
$Adam/m/lstm_106/lstm_cell_111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/m/lstm_106/lstm_cell_111/kernel
�
8Adam/m/lstm_106/lstm_cell_111/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_106/lstm_cell_111/kernel*
_output_shapes

:x*
dtype0
�
"Adam/v/lstm_105/lstm_cell_110/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/v/lstm_105/lstm_cell_110/bias
�
6Adam/v/lstm_105/lstm_cell_110/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_105/lstm_cell_110/bias*
_output_shapes
:x*
dtype0
�
"Adam/m/lstm_105/lstm_cell_110/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*3
shared_name$"Adam/m/lstm_105/lstm_cell_110/bias
�
6Adam/m/lstm_105/lstm_cell_110/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_105/lstm_cell_110/bias*
_output_shapes
:x*
dtype0
�
.Adam/v/lstm_105/lstm_cell_110/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/v/lstm_105/lstm_cell_110/recurrent_kernel
�
BAdam/v/lstm_105/lstm_cell_110/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_105/lstm_cell_110/recurrent_kernel*
_output_shapes

:x*
dtype0
�
.Adam/m/lstm_105/lstm_cell_110/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*?
shared_name0.Adam/m/lstm_105/lstm_cell_110/recurrent_kernel
�
BAdam/m/lstm_105/lstm_cell_110/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_105/lstm_cell_110/recurrent_kernel*
_output_shapes

:x*
dtype0
�
$Adam/v/lstm_105/lstm_cell_110/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/v/lstm_105/lstm_cell_110/kernel
�
8Adam/v/lstm_105/lstm_cell_110/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_105/lstm_cell_110/kernel*
_output_shapes

:x*
dtype0
�
$Adam/m/lstm_105/lstm_cell_110/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*5
shared_name&$Adam/m/lstm_105/lstm_cell_110/kernel
�
8Adam/m/lstm_105/lstm_cell_110/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_105/lstm_cell_110/kernel*
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
lstm_107/lstm_cell_112/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namelstm_107/lstm_cell_112/bias
�
/lstm_107/lstm_cell_112/bias/Read/ReadVariableOpReadVariableOplstm_107/lstm_cell_112/bias*
_output_shapes
:x*
dtype0
�
'lstm_107/lstm_cell_112/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*8
shared_name)'lstm_107/lstm_cell_112/recurrent_kernel
�
;lstm_107/lstm_cell_112/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_107/lstm_cell_112/recurrent_kernel*
_output_shapes

:x*
dtype0
�
lstm_107/lstm_cell_112/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*.
shared_namelstm_107/lstm_cell_112/kernel
�
1lstm_107/lstm_cell_112/kernel/Read/ReadVariableOpReadVariableOplstm_107/lstm_cell_112/kernel*
_output_shapes

:x*
dtype0
�
lstm_106/lstm_cell_111/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namelstm_106/lstm_cell_111/bias
�
/lstm_106/lstm_cell_111/bias/Read/ReadVariableOpReadVariableOplstm_106/lstm_cell_111/bias*
_output_shapes
:x*
dtype0
�
'lstm_106/lstm_cell_111/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*8
shared_name)'lstm_106/lstm_cell_111/recurrent_kernel
�
;lstm_106/lstm_cell_111/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_106/lstm_cell_111/recurrent_kernel*
_output_shapes

:x*
dtype0
�
lstm_106/lstm_cell_111/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*.
shared_namelstm_106/lstm_cell_111/kernel
�
1lstm_106/lstm_cell_111/kernel/Read/ReadVariableOpReadVariableOplstm_106/lstm_cell_111/kernel*
_output_shapes

:x*
dtype0
�
lstm_105/lstm_cell_110/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*,
shared_namelstm_105/lstm_cell_110/bias
�
/lstm_105/lstm_cell_110/bias/Read/ReadVariableOpReadVariableOplstm_105/lstm_cell_110/bias*
_output_shapes
:x*
dtype0
�
'lstm_105/lstm_cell_110/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*8
shared_name)'lstm_105/lstm_cell_110/recurrent_kernel
�
;lstm_105/lstm_cell_110/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_105/lstm_cell_110/recurrent_kernel*
_output_shapes

:x*
dtype0
�
lstm_105/lstm_cell_110/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*.
shared_namelstm_105/lstm_cell_110/kernel
�
1lstm_105/lstm_cell_110/kernel/Read/ReadVariableOpReadVariableOplstm_105/lstm_cell_110/kernel*
_output_shapes

:x*
dtype0
r
dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_85/bias
k
!dense_85/bias/Read/ReadVariableOpReadVariableOpdense_85/bias*
_output_shapes
:*
dtype0
z
dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_85/kernel
s
#dense_85/kernel/Read/ReadVariableOpReadVariableOpdense_85/kernel*
_output_shapes

:*
dtype0
�
serving_default_lstm_105_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_105_inputlstm_105/lstm_cell_110/kernel'lstm_105/lstm_cell_110/recurrent_kernellstm_105/lstm_cell_110/biaslstm_106/lstm_cell_111/kernel'lstm_106/lstm_cell_111/recurrent_kernellstm_106/lstm_cell_111/biaslstm_107/lstm_cell_112/kernel'lstm_107/lstm_cell_112/recurrent_kernellstm_107/lstm_cell_112/biasdense_85/kerneldense_85/bias*
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
&__inference_signature_wrapper_23133088

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
VARIABLE_VALUEdense_85/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_85/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_105/lstm_cell_110/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_105/lstm_cell_110/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_105/lstm_cell_110/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_106/lstm_cell_111/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_106/lstm_cell_111/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_106/lstm_cell_111/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_107/lstm_cell_112/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_107/lstm_cell_112/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_107/lstm_cell_112/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE$Adam/m/lstm_105/lstm_cell_110/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/lstm_105/lstm_cell_110/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/m/lstm_105/lstm_cell_110/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/v/lstm_105/lstm_cell_110/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/lstm_105/lstm_cell_110/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_105/lstm_cell_110/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/lstm_106/lstm_cell_111/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/lstm_106/lstm_cell_111/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/m/lstm_106/lstm_cell_111/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/lstm_106/lstm_cell_111/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/lstm_106/lstm_cell_111/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/lstm_106/lstm_cell_111/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/lstm_107/lstm_cell_112/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/lstm_107/lstm_cell_112/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/m/lstm_107/lstm_cell_112/recurrent_kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/lstm_107/lstm_cell_112/recurrent_kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/lstm_107/lstm_cell_112/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/lstm_107/lstm_cell_112/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_85/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_85/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_85/bias2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_85/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_85/kernel/Read/ReadVariableOp!dense_85/bias/Read/ReadVariableOp1lstm_105/lstm_cell_110/kernel/Read/ReadVariableOp;lstm_105/lstm_cell_110/recurrent_kernel/Read/ReadVariableOp/lstm_105/lstm_cell_110/bias/Read/ReadVariableOp1lstm_106/lstm_cell_111/kernel/Read/ReadVariableOp;lstm_106/lstm_cell_111/recurrent_kernel/Read/ReadVariableOp/lstm_106/lstm_cell_111/bias/Read/ReadVariableOp1lstm_107/lstm_cell_112/kernel/Read/ReadVariableOp;lstm_107/lstm_cell_112/recurrent_kernel/Read/ReadVariableOp/lstm_107/lstm_cell_112/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp8Adam/m/lstm_105/lstm_cell_110/kernel/Read/ReadVariableOp8Adam/v/lstm_105/lstm_cell_110/kernel/Read/ReadVariableOpBAdam/m/lstm_105/lstm_cell_110/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_105/lstm_cell_110/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_105/lstm_cell_110/bias/Read/ReadVariableOp6Adam/v/lstm_105/lstm_cell_110/bias/Read/ReadVariableOp8Adam/m/lstm_106/lstm_cell_111/kernel/Read/ReadVariableOp8Adam/v/lstm_106/lstm_cell_111/kernel/Read/ReadVariableOpBAdam/m/lstm_106/lstm_cell_111/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_106/lstm_cell_111/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_106/lstm_cell_111/bias/Read/ReadVariableOp6Adam/v/lstm_106/lstm_cell_111/bias/Read/ReadVariableOp8Adam/m/lstm_107/lstm_cell_112/kernel/Read/ReadVariableOp8Adam/v/lstm_107/lstm_cell_112/kernel/Read/ReadVariableOpBAdam/m/lstm_107/lstm_cell_112/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_107/lstm_cell_112/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_107/lstm_cell_112/bias/Read/ReadVariableOp6Adam/v/lstm_107/lstm_cell_112/bias/Read/ReadVariableOp*Adam/m/dense_85/kernel/Read/ReadVariableOp*Adam/v/dense_85/kernel/Read/ReadVariableOp(Adam/m/dense_85/bias/Read/ReadVariableOp(Adam/v/dense_85/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*4
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
!__inference__traced_save_23136345
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_85/kerneldense_85/biaslstm_105/lstm_cell_110/kernel'lstm_105/lstm_cell_110/recurrent_kernellstm_105/lstm_cell_110/biaslstm_106/lstm_cell_111/kernel'lstm_106/lstm_cell_111/recurrent_kernellstm_106/lstm_cell_111/biaslstm_107/lstm_cell_112/kernel'lstm_107/lstm_cell_112/recurrent_kernellstm_107/lstm_cell_112/bias	iterationlearning_rate$Adam/m/lstm_105/lstm_cell_110/kernel$Adam/v/lstm_105/lstm_cell_110/kernel.Adam/m/lstm_105/lstm_cell_110/recurrent_kernel.Adam/v/lstm_105/lstm_cell_110/recurrent_kernel"Adam/m/lstm_105/lstm_cell_110/bias"Adam/v/lstm_105/lstm_cell_110/bias$Adam/m/lstm_106/lstm_cell_111/kernel$Adam/v/lstm_106/lstm_cell_111/kernel.Adam/m/lstm_106/lstm_cell_111/recurrent_kernel.Adam/v/lstm_106/lstm_cell_111/recurrent_kernel"Adam/m/lstm_106/lstm_cell_111/bias"Adam/v/lstm_106/lstm_cell_111/bias$Adam/m/lstm_107/lstm_cell_112/kernel$Adam/v/lstm_107/lstm_cell_112/kernel.Adam/m/lstm_107/lstm_cell_112/recurrent_kernel.Adam/v/lstm_107/lstm_cell_112/recurrent_kernel"Adam/m/lstm_107/lstm_cell_112/bias"Adam/v/lstm_107/lstm_cell_112/biasAdam/m/dense_85/kernelAdam/v/dense_85/kernelAdam/m/dense_85/biasAdam/v/dense_85/biastotal_1count_1totalcount*3
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
$__inference__traced_restore_23136472��,
�D
�

lstm_107_while_body_23133480.
*lstm_107_while_lstm_107_while_loop_counter4
0lstm_107_while_lstm_107_while_maximum_iterations
lstm_107_while_placeholder 
lstm_107_while_placeholder_1 
lstm_107_while_placeholder_2 
lstm_107_while_placeholder_3-
)lstm_107_while_lstm_107_strided_slice_1_0i
elstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0:xQ
?lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0:xL
>lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0:x
lstm_107_while_identity
lstm_107_while_identity_1
lstm_107_while_identity_2
lstm_107_while_identity_3
lstm_107_while_identity_4
lstm_107_while_identity_5+
'lstm_107_while_lstm_107_strided_slice_1g
clstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensorM
;lstm_107_while_lstm_cell_112_matmul_readvariableop_resource:xO
=lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource:xJ
<lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource:x��3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp�2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp�4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp�
@lstm_107/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_107/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensor_0lstm_107_while_placeholderIlstm_107/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp=lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_107/while/lstm_cell_112/MatMulMatMul9lstm_107/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp?lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_107/while/lstm_cell_112/MatMul_1MatMullstm_107_while_placeholder_2<lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_107/while/lstm_cell_112/addAddV2-lstm_107/while/lstm_cell_112/MatMul:product:0/lstm_107/while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp>lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_107/while/lstm_cell_112/BiasAddBiasAdd$lstm_107/while/lstm_cell_112/add:z:0;lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_107/while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_107/while/lstm_cell_112/splitSplit5lstm_107/while/lstm_cell_112/split/split_dim:output:0-lstm_107/while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_107/while/lstm_cell_112/SigmoidSigmoid+lstm_107/while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:����������
&lstm_107/while/lstm_cell_112/Sigmoid_1Sigmoid+lstm_107/while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
 lstm_107/while/lstm_cell_112/mulMul*lstm_107/while/lstm_cell_112/Sigmoid_1:y:0lstm_107_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_107/while/lstm_cell_112/ReluRelu+lstm_107/while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
"lstm_107/while/lstm_cell_112/mul_1Mul(lstm_107/while/lstm_cell_112/Sigmoid:y:0/lstm_107/while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_107/while/lstm_cell_112/add_1AddV2$lstm_107/while/lstm_cell_112/mul:z:0&lstm_107/while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_107/while/lstm_cell_112/Sigmoid_2Sigmoid+lstm_107/while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:����������
#lstm_107/while/lstm_cell_112/Relu_1Relu&lstm_107/while/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_107/while/lstm_cell_112/mul_2Mul*lstm_107/while/lstm_cell_112/Sigmoid_2:y:01lstm_107/while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������{
9lstm_107/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
3lstm_107/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_107_while_placeholder_1Blstm_107/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_107/while/lstm_cell_112/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_107/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_107/while/addAddV2lstm_107_while_placeholderlstm_107/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_107/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/while/add_1AddV2*lstm_107_while_lstm_107_while_loop_counterlstm_107/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_107/while/IdentityIdentitylstm_107/while/add_1:z:0^lstm_107/while/NoOp*
T0*
_output_shapes
: �
lstm_107/while/Identity_1Identity0lstm_107_while_lstm_107_while_maximum_iterations^lstm_107/while/NoOp*
T0*
_output_shapes
: t
lstm_107/while/Identity_2Identitylstm_107/while/add:z:0^lstm_107/while/NoOp*
T0*
_output_shapes
: �
lstm_107/while/Identity_3IdentityClstm_107/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_107/while/NoOp*
T0*
_output_shapes
: �
lstm_107/while/Identity_4Identity&lstm_107/while/lstm_cell_112/mul_2:z:0^lstm_107/while/NoOp*
T0*'
_output_shapes
:����������
lstm_107/while/Identity_5Identity&lstm_107/while/lstm_cell_112/add_1:z:0^lstm_107/while/NoOp*
T0*'
_output_shapes
:����������
lstm_107/while/NoOpNoOp4^lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp3^lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp5^lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_107_while_identity lstm_107/while/Identity:output:0"?
lstm_107_while_identity_1"lstm_107/while/Identity_1:output:0"?
lstm_107_while_identity_2"lstm_107/while/Identity_2:output:0"?
lstm_107_while_identity_3"lstm_107/while/Identity_3:output:0"?
lstm_107_while_identity_4"lstm_107/while/Identity_4:output:0"?
lstm_107_while_identity_5"lstm_107/while/Identity_5:output:0"T
'lstm_107_while_lstm_107_strided_slice_1)lstm_107_while_lstm_107_strided_slice_1_0"~
<lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource>lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0"�
=lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource?lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0"|
;lstm_107_while_lstm_cell_112_matmul_readvariableop_resource=lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0"�
clstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensorelstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp2h
2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp2l
4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
while_body_23135490
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_112_matmul_readvariableop_resource_0:xH
6while_lstm_cell_112_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_112_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_112_matmul_readvariableop_resource:xF
4while_lstm_cell_112_matmul_1_readvariableop_resource:xA
3while_lstm_cell_112_biasadd_readvariableop_resource:x��*while/lstm_cell_112/BiasAdd/ReadVariableOp�)while/lstm_cell_112/MatMul/ReadVariableOp�+while/lstm_cell_112/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_112/addAddV2$while/lstm_cell_112/MatMul:product:0&while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_112/BiasAddBiasAddwhile/lstm_cell_112/add:z:02while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_112/splitSplit,while/lstm_cell_112/split/split_dim:output:0$while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_112/SigmoidSigmoid"while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_1Sigmoid"while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mulMul!while/lstm_cell_112/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_112/ReluRelu"while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_1Mulwhile/lstm_cell_112/Sigmoid:y:0&while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/add_1AddV2while/lstm_cell_112/mul:z:0while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_2Sigmoid"while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_112/Relu_1Reluwhile/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_2Mul!while/lstm_cell_112/Sigmoid_2:y:0(while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_112/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_112/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_112/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_112/BiasAdd/ReadVariableOp*^while/lstm_cell_112/MatMul/ReadVariableOp,^while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_112_biasadd_readvariableop_resource5while_lstm_cell_112_biasadd_readvariableop_resource_0"n
4while_lstm_cell_112_matmul_1_readvariableop_resource6while_lstm_cell_112_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_112_matmul_readvariableop_resource4while_lstm_cell_112_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_112/BiasAdd/ReadVariableOp*while/lstm_cell_112/BiasAdd/ReadVariableOp2V
)while/lstm_cell_112/MatMul/ReadVariableOp)while/lstm_cell_112/MatMul/ReadVariableOp2Z
+while/lstm_cell_112/MatMul_1/ReadVariableOp+while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
0__inference_sequential_87_layer_call_fn_23132353
lstm_105_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_105_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23132328o
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
_user_specified_namelstm_105_input
�

�
lstm_105_while_cond_23133630.
*lstm_105_while_lstm_105_while_loop_counter4
0lstm_105_while_lstm_105_while_maximum_iterations
lstm_105_while_placeholder 
lstm_105_while_placeholder_1 
lstm_105_while_placeholder_2 
lstm_105_while_placeholder_30
,lstm_105_while_less_lstm_105_strided_slice_1H
Dlstm_105_while_lstm_105_while_cond_23133630___redundant_placeholder0H
Dlstm_105_while_lstm_105_while_cond_23133630___redundant_placeholder1H
Dlstm_105_while_lstm_105_while_cond_23133630___redundant_placeholder2H
Dlstm_105_while_lstm_105_while_cond_23133630___redundant_placeholder3
lstm_105_while_identity
�
lstm_105/while/LessLesslstm_105_while_placeholder,lstm_105_while_less_lstm_105_strided_slice_1*
T0*
_output_shapes
: ]
lstm_105/while/IdentityIdentitylstm_105/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_105_while_identity lstm_105/while/Identity:output:0*(
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
�
�
+__inference_dense_85_layer_call_fn_23135901

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
F__inference_dense_85_layer_call_and_return_conditional_losses_23132321o
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23136009

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
�
F__inference_dense_85_layer_call_and_return_conditional_losses_23132321

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
�J
�
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132144

inputs>
,lstm_cell_111_matmul_readvariableop_resource:x@
.lstm_cell_111_matmul_1_readvariableop_resource:x;
-lstm_cell_111_biasadd_readvariableop_resource:x
identity��$lstm_cell_111/BiasAdd/ReadVariableOp�#lstm_cell_111/MatMul/ReadVariableOp�%lstm_cell_111/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_111/MatMul/ReadVariableOpReadVariableOp,lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMulMatMulstrided_slice_2:output:0+lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMul_1MatMulzeros:output:0-lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_111/addAddV2lstm_cell_111/MatMul:product:0 lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_111/BiasAddBiasAddlstm_cell_111/add:z:0,lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_111/splitSplit&lstm_cell_111/split/split_dim:output:0lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_111/SigmoidSigmoidlstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_1Sigmoidlstm_cell_111/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_111/mulMullstm_cell_111/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_111/ReluRelulstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_1Mullstm_cell_111/Sigmoid:y:0 lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_111/add_1AddV2lstm_cell_111/mul:z:0lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_2Sigmoidlstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_111/Relu_1Relulstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_2Mullstm_cell_111/Sigmoid_2:y:0"lstm_cell_111/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_111_matmul_readvariableop_resource.lstm_cell_111_matmul_1_readvariableop_resource-lstm_cell_111_biasadd_readvariableop_resource*
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
while_body_23132060*
condR
while_cond_23132059*K
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
NoOpNoOp%^lstm_cell_111/BiasAdd/ReadVariableOp$^lstm_cell_111/MatMul/ReadVariableOp&^lstm_cell_111/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_111/BiasAdd/ReadVariableOp$lstm_cell_111/BiasAdd/ReadVariableOp2J
#lstm_cell_111/MatMul/ReadVariableOp#lstm_cell_111/MatMul/ReadVariableOp2N
%lstm_cell_111/MatMul_1/ReadVariableOp%lstm_cell_111/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_23134255
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_110_matmul_readvariableop_resource_0:xH
6while_lstm_cell_110_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_110_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_110_matmul_readvariableop_resource:xF
4while_lstm_cell_110_matmul_1_readvariableop_resource:xA
3while_lstm_cell_110_biasadd_readvariableop_resource:x��*while/lstm_cell_110/BiasAdd/ReadVariableOp�)while/lstm_cell_110/MatMul/ReadVariableOp�+while/lstm_cell_110/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_110/addAddV2$while/lstm_cell_110/MatMul:product:0&while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_110/BiasAddBiasAddwhile/lstm_cell_110/add:z:02while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_110/splitSplit,while/lstm_cell_110/split/split_dim:output:0$while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_110/SigmoidSigmoid"while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_1Sigmoid"while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mulMul!while/lstm_cell_110/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_110/ReluRelu"while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_1Mulwhile/lstm_cell_110/Sigmoid:y:0&while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/add_1AddV2while/lstm_cell_110/mul:z:0while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_2Sigmoid"while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_110/Relu_1Reluwhile/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_2Mul!while/lstm_cell_110/Sigmoid_2:y:0(while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_110/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_110/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_110/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_110/BiasAdd/ReadVariableOp*^while/lstm_cell_110/MatMul/ReadVariableOp,^while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_110_biasadd_readvariableop_resource5while_lstm_cell_110_biasadd_readvariableop_resource_0"n
4while_lstm_cell_110_matmul_1_readvariableop_resource6while_lstm_cell_110_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_110_matmul_readvariableop_resource4while_lstm_cell_110_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_110/BiasAdd/ReadVariableOp*while/lstm_cell_110/BiasAdd/ReadVariableOp2V
)while/lstm_cell_110/MatMul/ReadVariableOp)while/lstm_cell_110/MatMul/ReadVariableOp2Z
+while/lstm_cell_110/MatMul_1/ReadVariableOp+while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23131004

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

g
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135892

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
while_body_23132459
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_112_matmul_readvariableop_resource_0:xH
6while_lstm_cell_112_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_112_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_112_matmul_readvariableop_resource:xF
4while_lstm_cell_112_matmul_1_readvariableop_resource:xA
3while_lstm_cell_112_biasadd_readvariableop_resource:x��*while/lstm_cell_112/BiasAdd/ReadVariableOp�)while/lstm_cell_112/MatMul/ReadVariableOp�+while/lstm_cell_112/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_112/addAddV2$while/lstm_cell_112/MatMul:product:0&while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_112/BiasAddBiasAddwhile/lstm_cell_112/add:z:02while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_112/splitSplit,while/lstm_cell_112/split/split_dim:output:0$while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_112/SigmoidSigmoid"while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_1Sigmoid"while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mulMul!while/lstm_cell_112/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_112/ReluRelu"while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_1Mulwhile/lstm_cell_112/Sigmoid:y:0&while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/add_1AddV2while/lstm_cell_112/mul:z:0while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_2Sigmoid"while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_112/Relu_1Reluwhile/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_2Mul!while/lstm_cell_112/Sigmoid_2:y:0(while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_112/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_112/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_112/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_112/BiasAdd/ReadVariableOp*^while/lstm_cell_112/MatMul/ReadVariableOp,^while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_112_biasadd_readvariableop_resource5while_lstm_cell_112_biasadd_readvariableop_resource_0"n
4while_lstm_cell_112_matmul_1_readvariableop_resource6while_lstm_cell_112_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_112_matmul_readvariableop_resource4while_lstm_cell_112_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_112/BiasAdd/ReadVariableOp*while/lstm_cell_112/BiasAdd/ReadVariableOp2V
)while/lstm_cell_112/MatMul/ReadVariableOp)while/lstm_cell_112/MatMul/ReadVariableOp2Z
+while/lstm_cell_112/MatMul_1/ReadVariableOp+while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
while_body_23131573
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_112_23131597_0:x0
while_lstm_cell_112_23131599_0:x,
while_lstm_cell_112_23131601_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_112_23131597:x.
while_lstm_cell_112_23131599:x*
while_lstm_cell_112_23131601:x��+while/lstm_cell_112/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_112/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_112_23131597_0while_lstm_cell_112_23131599_0while_lstm_cell_112_23131601_0*
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131558r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_112/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_112/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_112/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_112/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_112_23131597while_lstm_cell_112_23131597_0">
while_lstm_cell_112_23131599while_lstm_cell_112_23131599_0">
while_lstm_cell_112_23131601while_lstm_cell_112_23131601_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_112/StatefulPartitionedCall+while/lstm_cell_112/StatefulPartitionedCall: 
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
while_cond_23132624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23132624___redundant_placeholder06
2while_while_cond_23132624___redundant_placeholder16
2while_while_cond_23132624___redundant_placeholder26
2while_while_cond_23132624___redundant_placeholder3
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135720

inputs>
,lstm_cell_112_matmul_readvariableop_resource:x@
.lstm_cell_112_matmul_1_readvariableop_resource:x;
-lstm_cell_112_biasadd_readvariableop_resource:x
identity��$lstm_cell_112/BiasAdd/ReadVariableOp�#lstm_cell_112/MatMul/ReadVariableOp�%lstm_cell_112/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_112/MatMul/ReadVariableOpReadVariableOp,lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMulMatMulstrided_slice_2:output:0+lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMul_1MatMulzeros:output:0-lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_112/addAddV2lstm_cell_112/MatMul:product:0 lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_112/BiasAddBiasAddlstm_cell_112/add:z:0,lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_112/splitSplit&lstm_cell_112/split/split_dim:output:0lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_112/SigmoidSigmoidlstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_1Sigmoidlstm_cell_112/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_112/mulMullstm_cell_112/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_112/ReluRelulstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_1Mullstm_cell_112/Sigmoid:y:0 lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_112/add_1AddV2lstm_cell_112/mul:z:0lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_2Sigmoidlstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_112/Relu_1Relulstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_2Mullstm_cell_112/Sigmoid_2:y:0"lstm_cell_112/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_112_matmul_readvariableop_resource.lstm_cell_112_matmul_1_readvariableop_resource-lstm_cell_112_biasadd_readvariableop_resource*
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
while_body_23135635*
condR
while_cond_23135634*K
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
NoOpNoOp%^lstm_cell_112/BiasAdd/ReadVariableOp$^lstm_cell_112/MatMul/ReadVariableOp&^lstm_cell_112/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_112/BiasAdd/ReadVariableOp$lstm_cell_112/BiasAdd/ReadVariableOp2J
#lstm_cell_112/MatMul/ReadVariableOp#lstm_cell_112/MatMul/ReadVariableOp2N
%lstm_cell_112/MatMul_1/ReadVariableOp%lstm_cell_112/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134339
inputs_0>
,lstm_cell_110_matmul_readvariableop_resource:x@
.lstm_cell_110_matmul_1_readvariableop_resource:x;
-lstm_cell_110_biasadd_readvariableop_resource:x
identity��$lstm_cell_110/BiasAdd/ReadVariableOp�#lstm_cell_110/MatMul/ReadVariableOp�%lstm_cell_110/MatMul_1/ReadVariableOp�while=
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
#lstm_cell_110/MatMul/ReadVariableOpReadVariableOp,lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMulMatMulstrided_slice_2:output:0+lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMul_1MatMulzeros:output:0-lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_110/addAddV2lstm_cell_110/MatMul:product:0 lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_110/BiasAddBiasAddlstm_cell_110/add:z:0,lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_110/splitSplit&lstm_cell_110/split/split_dim:output:0lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_110/SigmoidSigmoidlstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_1Sigmoidlstm_cell_110/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_110/mulMullstm_cell_110/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_110/ReluRelulstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_1Mullstm_cell_110/Sigmoid:y:0 lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_110/add_1AddV2lstm_cell_110/mul:z:0lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_2Sigmoidlstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_110/Relu_1Relulstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_2Mullstm_cell_110/Sigmoid_2:y:0"lstm_cell_110/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_110_matmul_readvariableop_resource.lstm_cell_110_matmul_1_readvariableop_resource-lstm_cell_110_biasadd_readvariableop_resource*
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
while_body_23134255*
condR
while_cond_23134254*K
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
NoOpNoOp%^lstm_cell_110/BiasAdd/ReadVariableOp$^lstm_cell_110/MatMul/ReadVariableOp&^lstm_cell_110/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_110/BiasAdd/ReadVariableOp$lstm_cell_110/BiasAdd/ReadVariableOp2J
#lstm_cell_110/MatMul/ReadVariableOp#lstm_cell_110/MatMul/ReadVariableOp2N
%lstm_cell_110/MatMul_1/ReadVariableOp%lstm_cell_110/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�8
�
F__inference_lstm_105_layer_call_and_return_conditional_losses_23131132

inputs(
lstm_cell_110_23131050:x(
lstm_cell_110_23131052:x$
lstm_cell_110_23131054:x
identity��%lstm_cell_110/StatefulPartitionedCall�while;
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
%lstm_cell_110/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_110_23131050lstm_cell_110_23131052lstm_cell_110_23131054*
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23131004n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_110_23131050lstm_cell_110_23131052lstm_cell_110_23131054*
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
while_body_23131063*
condR
while_cond_23131062*K
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
NoOpNoOp&^lstm_cell_110/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_110/StatefulPartitionedCall%lstm_cell_110/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�

�
lstm_106_while_cond_23133769.
*lstm_106_while_lstm_106_while_loop_counter4
0lstm_106_while_lstm_106_while_maximum_iterations
lstm_106_while_placeholder 
lstm_106_while_placeholder_1 
lstm_106_while_placeholder_2 
lstm_106_while_placeholder_30
,lstm_106_while_less_lstm_106_strided_slice_1H
Dlstm_106_while_lstm_106_while_cond_23133769___redundant_placeholder0H
Dlstm_106_while_lstm_106_while_cond_23133769___redundant_placeholder1H
Dlstm_106_while_lstm_106_while_cond_23133769___redundant_placeholder2H
Dlstm_106_while_lstm_106_while_cond_23133769___redundant_placeholder3
lstm_106_while_identity
�
lstm_106/while/LessLesslstm_106_while_placeholder,lstm_106_while_less_lstm_106_strided_slice_1*
T0*
_output_shapes
: ]
lstm_106/while/IdentityIdentitylstm_106/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_106_while_identity lstm_106/while/Identity:output:0*(
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
+__inference_lstm_107_layer_call_fn_23135263
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23131836o
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
while_body_23134112
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_110_matmul_readvariableop_resource_0:xH
6while_lstm_cell_110_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_110_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_110_matmul_readvariableop_resource:xF
4while_lstm_cell_110_matmul_1_readvariableop_resource:xA
3while_lstm_cell_110_biasadd_readvariableop_resource:x��*while/lstm_cell_110/BiasAdd/ReadVariableOp�)while/lstm_cell_110/MatMul/ReadVariableOp�+while/lstm_cell_110/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_110/addAddV2$while/lstm_cell_110/MatMul:product:0&while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_110/BiasAddBiasAddwhile/lstm_cell_110/add:z:02while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_110/splitSplit,while/lstm_cell_110/split/split_dim:output:0$while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_110/SigmoidSigmoid"while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_1Sigmoid"while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mulMul!while/lstm_cell_110/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_110/ReluRelu"while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_1Mulwhile/lstm_cell_110/Sigmoid:y:0&while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/add_1AddV2while/lstm_cell_110/mul:z:0while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_2Sigmoid"while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_110/Relu_1Reluwhile/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_2Mul!while/lstm_cell_110/Sigmoid_2:y:0(while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_110/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_110/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_110/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_110/BiasAdd/ReadVariableOp*^while/lstm_cell_110/MatMul/ReadVariableOp,^while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_110_biasadd_readvariableop_resource5while_lstm_cell_110_biasadd_readvariableop_resource_0"n
4while_lstm_cell_110_matmul_1_readvariableop_resource6while_lstm_cell_110_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_110_matmul_readvariableop_resource4while_lstm_cell_110_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_110/BiasAdd/ReadVariableOp*while/lstm_cell_110/BiasAdd/ReadVariableOp2V
)while/lstm_cell_110/MatMul/ReadVariableOp)while/lstm_cell_110/MatMul/ReadVariableOp2Z
+while/lstm_cell_110/MatMul_1/ReadVariableOp+while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
while_cond_23132059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23132059___redundant_placeholder06
2while_while_cond_23132059___redundant_placeholder16
2while_while_cond_23132059___redundant_placeholder26
2while_while_cond_23132059___redundant_placeholder3
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134196
inputs_0>
,lstm_cell_110_matmul_readvariableop_resource:x@
.lstm_cell_110_matmul_1_readvariableop_resource:x;
-lstm_cell_110_biasadd_readvariableop_resource:x
identity��$lstm_cell_110/BiasAdd/ReadVariableOp�#lstm_cell_110/MatMul/ReadVariableOp�%lstm_cell_110/MatMul_1/ReadVariableOp�while=
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
#lstm_cell_110/MatMul/ReadVariableOpReadVariableOp,lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMulMatMulstrided_slice_2:output:0+lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMul_1MatMulzeros:output:0-lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_110/addAddV2lstm_cell_110/MatMul:product:0 lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_110/BiasAddBiasAddlstm_cell_110/add:z:0,lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_110/splitSplit&lstm_cell_110/split/split_dim:output:0lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_110/SigmoidSigmoidlstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_1Sigmoidlstm_cell_110/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_110/mulMullstm_cell_110/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_110/ReluRelulstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_1Mullstm_cell_110/Sigmoid:y:0 lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_110/add_1AddV2lstm_cell_110/mul:z:0lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_2Sigmoidlstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_110/Relu_1Relulstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_2Mullstm_cell_110/Sigmoid_2:y:0"lstm_cell_110/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_110_matmul_readvariableop_resource.lstm_cell_110_matmul_1_readvariableop_resource-lstm_cell_110_biasadd_readvariableop_resource*
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
while_body_23134112*
condR
while_cond_23134111*K
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
NoOpNoOp%^lstm_cell_110/BiasAdd/ReadVariableOp$^lstm_cell_110/MatMul/ReadVariableOp&^lstm_cell_110/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_110/BiasAdd/ReadVariableOp$lstm_cell_110/BiasAdd/ReadVariableOp2J
#lstm_cell_110/MatMul/ReadVariableOp#lstm_cell_110/MatMul/ReadVariableOp2N
%lstm_cell_110/MatMul_1/ReadVariableOp%lstm_cell_110/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�S
�
!__inference__traced_save_23136345
file_prefix.
*savev2_dense_85_kernel_read_readvariableop,
(savev2_dense_85_bias_read_readvariableop<
8savev2_lstm_105_lstm_cell_110_kernel_read_readvariableopF
Bsavev2_lstm_105_lstm_cell_110_recurrent_kernel_read_readvariableop:
6savev2_lstm_105_lstm_cell_110_bias_read_readvariableop<
8savev2_lstm_106_lstm_cell_111_kernel_read_readvariableopF
Bsavev2_lstm_106_lstm_cell_111_recurrent_kernel_read_readvariableop:
6savev2_lstm_106_lstm_cell_111_bias_read_readvariableop<
8savev2_lstm_107_lstm_cell_112_kernel_read_readvariableopF
Bsavev2_lstm_107_lstm_cell_112_recurrent_kernel_read_readvariableop:
6savev2_lstm_107_lstm_cell_112_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopC
?savev2_adam_m_lstm_105_lstm_cell_110_kernel_read_readvariableopC
?savev2_adam_v_lstm_105_lstm_cell_110_kernel_read_readvariableopM
Isavev2_adam_m_lstm_105_lstm_cell_110_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_105_lstm_cell_110_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_105_lstm_cell_110_bias_read_readvariableopA
=savev2_adam_v_lstm_105_lstm_cell_110_bias_read_readvariableopC
?savev2_adam_m_lstm_106_lstm_cell_111_kernel_read_readvariableopC
?savev2_adam_v_lstm_106_lstm_cell_111_kernel_read_readvariableopM
Isavev2_adam_m_lstm_106_lstm_cell_111_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_106_lstm_cell_111_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_106_lstm_cell_111_bias_read_readvariableopA
=savev2_adam_v_lstm_106_lstm_cell_111_bias_read_readvariableopC
?savev2_adam_m_lstm_107_lstm_cell_112_kernel_read_readvariableopC
?savev2_adam_v_lstm_107_lstm_cell_112_kernel_read_readvariableopM
Isavev2_adam_m_lstm_107_lstm_cell_112_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_107_lstm_cell_112_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_107_lstm_cell_112_bias_read_readvariableopA
=savev2_adam_v_lstm_107_lstm_cell_112_bias_read_readvariableop5
1savev2_adam_m_dense_85_kernel_read_readvariableop5
1savev2_adam_v_dense_85_kernel_read_readvariableop3
/savev2_adam_m_dense_85_bias_read_readvariableop3
/savev2_adam_v_dense_85_bias_read_readvariableop&
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_85_kernel_read_readvariableop(savev2_dense_85_bias_read_readvariableop8savev2_lstm_105_lstm_cell_110_kernel_read_readvariableopBsavev2_lstm_105_lstm_cell_110_recurrent_kernel_read_readvariableop6savev2_lstm_105_lstm_cell_110_bias_read_readvariableop8savev2_lstm_106_lstm_cell_111_kernel_read_readvariableopBsavev2_lstm_106_lstm_cell_111_recurrent_kernel_read_readvariableop6savev2_lstm_106_lstm_cell_111_bias_read_readvariableop8savev2_lstm_107_lstm_cell_112_kernel_read_readvariableopBsavev2_lstm_107_lstm_cell_112_recurrent_kernel_read_readvariableop6savev2_lstm_107_lstm_cell_112_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop?savev2_adam_m_lstm_105_lstm_cell_110_kernel_read_readvariableop?savev2_adam_v_lstm_105_lstm_cell_110_kernel_read_readvariableopIsavev2_adam_m_lstm_105_lstm_cell_110_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_105_lstm_cell_110_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_105_lstm_cell_110_bias_read_readvariableop=savev2_adam_v_lstm_105_lstm_cell_110_bias_read_readvariableop?savev2_adam_m_lstm_106_lstm_cell_111_kernel_read_readvariableop?savev2_adam_v_lstm_106_lstm_cell_111_kernel_read_readvariableopIsavev2_adam_m_lstm_106_lstm_cell_111_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_106_lstm_cell_111_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_106_lstm_cell_111_bias_read_readvariableop=savev2_adam_v_lstm_106_lstm_cell_111_bias_read_readvariableop?savev2_adam_m_lstm_107_lstm_cell_112_kernel_read_readvariableop?savev2_adam_v_lstm_107_lstm_cell_112_kernel_read_readvariableopIsavev2_adam_m_lstm_107_lstm_cell_112_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_107_lstm_cell_112_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_107_lstm_cell_112_bias_read_readvariableop=savev2_adam_v_lstm_107_lstm_cell_112_bias_read_readvariableop1savev2_adam_m_dense_85_kernel_read_readvariableop1savev2_adam_v_dense_85_kernel_read_readvariableop/savev2_adam_m_dense_85_bias_read_readvariableop/savev2_adam_v_dense_85_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
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
while_cond_23131909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23131909___redundant_placeholder06
2while_while_cond_23131909___redundant_placeholder16
2while_while_cond_23131909___redundant_placeholder26
2while_while_cond_23131909___redundant_placeholder3
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
while_body_23131413
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_111_23131437_0:x0
while_lstm_cell_111_23131439_0:x,
while_lstm_cell_111_23131441_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_111_23131437:x.
while_lstm_cell_111_23131439:x*
while_lstm_cell_111_23131441:x��+while/lstm_cell_111/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_111/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_111_23131437_0while_lstm_cell_111_23131439_0while_lstm_cell_111_23131441_0*
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131354�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_111/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_111/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_111/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_111/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_111_23131437while_lstm_cell_111_23131437_0">
while_lstm_cell_111_23131439while_lstm_cell_111_23131439_0">
while_lstm_cell_111_23131441while_lstm_cell_111_23131441_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_111/StatefulPartitionedCall+while/lstm_cell_111/StatefulPartitionedCall: 
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
while_cond_23130871
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23130871___redundant_placeholder06
2while_while_cond_23130871___redundant_placeholder16
2while_while_cond_23130871___redundant_placeholder26
2while_while_cond_23130871___redundant_placeholder3
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134812
inputs_0>
,lstm_cell_111_matmul_readvariableop_resource:x@
.lstm_cell_111_matmul_1_readvariableop_resource:x;
-lstm_cell_111_biasadd_readvariableop_resource:x
identity��$lstm_cell_111/BiasAdd/ReadVariableOp�#lstm_cell_111/MatMul/ReadVariableOp�%lstm_cell_111/MatMul_1/ReadVariableOp�while=
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
#lstm_cell_111/MatMul/ReadVariableOpReadVariableOp,lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMulMatMulstrided_slice_2:output:0+lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMul_1MatMulzeros:output:0-lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_111/addAddV2lstm_cell_111/MatMul:product:0 lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_111/BiasAddBiasAddlstm_cell_111/add:z:0,lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_111/splitSplit&lstm_cell_111/split/split_dim:output:0lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_111/SigmoidSigmoidlstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_1Sigmoidlstm_cell_111/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_111/mulMullstm_cell_111/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_111/ReluRelulstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_1Mullstm_cell_111/Sigmoid:y:0 lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_111/add_1AddV2lstm_cell_111/mul:z:0lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_2Sigmoidlstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_111/Relu_1Relulstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_2Mullstm_cell_111/Sigmoid_2:y:0"lstm_cell_111/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_111_matmul_readvariableop_resource.lstm_cell_111_matmul_1_readvariableop_resource-lstm_cell_111_biasadd_readvariableop_resource*
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
while_body_23134728*
condR
while_cond_23134727*K
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
NoOpNoOp%^lstm_cell_111/BiasAdd/ReadVariableOp$^lstm_cell_111/MatMul/ReadVariableOp&^lstm_cell_111/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_111/BiasAdd/ReadVariableOp$lstm_cell_111/BiasAdd/ReadVariableOp2J
#lstm_cell_111/MatMul/ReadVariableOp#lstm_cell_111/MatMul/ReadVariableOp2N
%lstm_cell_111/MatMul_1/ReadVariableOp%lstm_cell_111/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�8
�
while_body_23131910
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_110_matmul_readvariableop_resource_0:xH
6while_lstm_cell_110_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_110_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_110_matmul_readvariableop_resource:xF
4while_lstm_cell_110_matmul_1_readvariableop_resource:xA
3while_lstm_cell_110_biasadd_readvariableop_resource:x��*while/lstm_cell_110/BiasAdd/ReadVariableOp�)while/lstm_cell_110/MatMul/ReadVariableOp�+while/lstm_cell_110/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_110/addAddV2$while/lstm_cell_110/MatMul:product:0&while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_110/BiasAddBiasAddwhile/lstm_cell_110/add:z:02while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_110/splitSplit,while/lstm_cell_110/split/split_dim:output:0$while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_110/SigmoidSigmoid"while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_1Sigmoid"while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mulMul!while/lstm_cell_110/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_110/ReluRelu"while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_1Mulwhile/lstm_cell_110/Sigmoid:y:0&while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/add_1AddV2while/lstm_cell_110/mul:z:0while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_2Sigmoid"while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_110/Relu_1Reluwhile/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_2Mul!while/lstm_cell_110/Sigmoid_2:y:0(while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_110/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_110/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_110/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_110/BiasAdd/ReadVariableOp*^while/lstm_cell_110/MatMul/ReadVariableOp,^while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_110_biasadd_readvariableop_resource5while_lstm_cell_110_biasadd_readvariableop_resource_0"n
4while_lstm_cell_110_matmul_1_readvariableop_resource6while_lstm_cell_110_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_110_matmul_readvariableop_resource4while_lstm_cell_110_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_110/BiasAdd/ReadVariableOp*while/lstm_cell_110/BiasAdd/ReadVariableOp2V
)while/lstm_cell_110/MatMul/ReadVariableOp)while/lstm_cell_110/MatMul/ReadVariableOp2Z
+while/lstm_cell_110/MatMul_1/ReadVariableOp+while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
while_body_23130872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_110_23130896_0:x0
while_lstm_cell_110_23130898_0:x,
while_lstm_cell_110_23130900_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_110_23130896:x.
while_lstm_cell_110_23130898:x*
while_lstm_cell_110_23130900:x��+while/lstm_cell_110/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_110/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_110_23130896_0while_lstm_cell_110_23130898_0while_lstm_cell_110_23130900_0*
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23130858�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_110/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_110/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_110/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_110/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_110_23130896while_lstm_cell_110_23130896_0">
while_lstm_cell_110_23130898while_lstm_cell_110_23130898_0">
while_lstm_cell_110_23130900while_lstm_cell_110_23130900_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_110/StatefulPartitionedCall+while/lstm_cell_110/StatefulPartitionedCall: 
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

lstm_106_while_body_23133340.
*lstm_106_while_lstm_106_while_loop_counter4
0lstm_106_while_lstm_106_while_maximum_iterations
lstm_106_while_placeholder 
lstm_106_while_placeholder_1 
lstm_106_while_placeholder_2 
lstm_106_while_placeholder_3-
)lstm_106_while_lstm_106_strided_slice_1_0i
elstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0:xQ
?lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0:xL
>lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0:x
lstm_106_while_identity
lstm_106_while_identity_1
lstm_106_while_identity_2
lstm_106_while_identity_3
lstm_106_while_identity_4
lstm_106_while_identity_5+
'lstm_106_while_lstm_106_strided_slice_1g
clstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensorM
;lstm_106_while_lstm_cell_111_matmul_readvariableop_resource:xO
=lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource:xJ
<lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource:x��3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp�2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp�4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp�
@lstm_106/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_106/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensor_0lstm_106_while_placeholderIlstm_106/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp=lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_106/while/lstm_cell_111/MatMulMatMul9lstm_106/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp?lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_106/while/lstm_cell_111/MatMul_1MatMullstm_106_while_placeholder_2<lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_106/while/lstm_cell_111/addAddV2-lstm_106/while/lstm_cell_111/MatMul:product:0/lstm_106/while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp>lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_106/while/lstm_cell_111/BiasAddBiasAdd$lstm_106/while/lstm_cell_111/add:z:0;lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_106/while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_106/while/lstm_cell_111/splitSplit5lstm_106/while/lstm_cell_111/split/split_dim:output:0-lstm_106/while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_106/while/lstm_cell_111/SigmoidSigmoid+lstm_106/while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:����������
&lstm_106/while/lstm_cell_111/Sigmoid_1Sigmoid+lstm_106/while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
 lstm_106/while/lstm_cell_111/mulMul*lstm_106/while/lstm_cell_111/Sigmoid_1:y:0lstm_106_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_106/while/lstm_cell_111/ReluRelu+lstm_106/while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
"lstm_106/while/lstm_cell_111/mul_1Mul(lstm_106/while/lstm_cell_111/Sigmoid:y:0/lstm_106/while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_106/while/lstm_cell_111/add_1AddV2$lstm_106/while/lstm_cell_111/mul:z:0&lstm_106/while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_106/while/lstm_cell_111/Sigmoid_2Sigmoid+lstm_106/while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:����������
#lstm_106/while/lstm_cell_111/Relu_1Relu&lstm_106/while/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_106/while/lstm_cell_111/mul_2Mul*lstm_106/while/lstm_cell_111/Sigmoid_2:y:01lstm_106/while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_106/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_106_while_placeholder_1lstm_106_while_placeholder&lstm_106/while/lstm_cell_111/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_106/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_106/while/addAddV2lstm_106_while_placeholderlstm_106/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_106/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/while/add_1AddV2*lstm_106_while_lstm_106_while_loop_counterlstm_106/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_106/while/IdentityIdentitylstm_106/while/add_1:z:0^lstm_106/while/NoOp*
T0*
_output_shapes
: �
lstm_106/while/Identity_1Identity0lstm_106_while_lstm_106_while_maximum_iterations^lstm_106/while/NoOp*
T0*
_output_shapes
: t
lstm_106/while/Identity_2Identitylstm_106/while/add:z:0^lstm_106/while/NoOp*
T0*
_output_shapes
: �
lstm_106/while/Identity_3IdentityClstm_106/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_106/while/NoOp*
T0*
_output_shapes
: �
lstm_106/while/Identity_4Identity&lstm_106/while/lstm_cell_111/mul_2:z:0^lstm_106/while/NoOp*
T0*'
_output_shapes
:����������
lstm_106/while/Identity_5Identity&lstm_106/while/lstm_cell_111/add_1:z:0^lstm_106/while/NoOp*
T0*'
_output_shapes
:����������
lstm_106/while/NoOpNoOp4^lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp3^lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp5^lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_106_while_identity lstm_106/while/Identity:output:0"?
lstm_106_while_identity_1"lstm_106/while/Identity_1:output:0"?
lstm_106_while_identity_2"lstm_106/while/Identity_2:output:0"?
lstm_106_while_identity_3"lstm_106/while/Identity_3:output:0"?
lstm_106_while_identity_4"lstm_106/while/Identity_4:output:0"?
lstm_106_while_identity_5"lstm_106/while/Identity_5:output:0"T
'lstm_106_while_lstm_106_strided_slice_1)lstm_106_while_lstm_106_strided_slice_1_0"~
<lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource>lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0"�
=lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource?lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0"|
;lstm_106_while_lstm_cell_111_matmul_readvariableop_resource=lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0"�
clstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensorelstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp2h
2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp2l
4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
F__inference_dense_85_layer_call_and_return_conditional_losses_23135911

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
�
�
+__inference_lstm_106_layer_call_fn_23134636
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23131291|
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
�J
�
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132709

inputs>
,lstm_cell_111_matmul_readvariableop_resource:x@
.lstm_cell_111_matmul_1_readvariableop_resource:x;
-lstm_cell_111_biasadd_readvariableop_resource:x
identity��$lstm_cell_111/BiasAdd/ReadVariableOp�#lstm_cell_111/MatMul/ReadVariableOp�%lstm_cell_111/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_111/MatMul/ReadVariableOpReadVariableOp,lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMulMatMulstrided_slice_2:output:0+lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMul_1MatMulzeros:output:0-lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_111/addAddV2lstm_cell_111/MatMul:product:0 lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_111/BiasAddBiasAddlstm_cell_111/add:z:0,lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_111/splitSplit&lstm_cell_111/split/split_dim:output:0lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_111/SigmoidSigmoidlstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_1Sigmoidlstm_cell_111/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_111/mulMullstm_cell_111/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_111/ReluRelulstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_1Mullstm_cell_111/Sigmoid:y:0 lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_111/add_1AddV2lstm_cell_111/mul:z:0lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_2Sigmoidlstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_111/Relu_1Relulstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_2Mullstm_cell_111/Sigmoid_2:y:0"lstm_cell_111/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_111_matmul_readvariableop_resource.lstm_cell_111_matmul_1_readvariableop_resource-lstm_cell_111_biasadd_readvariableop_resource*
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
while_body_23132625*
condR
while_cond_23132624*K
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
NoOpNoOp%^lstm_cell_111/BiasAdd/ReadVariableOp$^lstm_cell_111/MatMul/ReadVariableOp&^lstm_cell_111/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_111/BiasAdd/ReadVariableOp$lstm_cell_111/BiasAdd/ReadVariableOp2J
#lstm_cell_111/MatMul/ReadVariableOp#lstm_cell_111/MatMul/ReadVariableOp2N
%lstm_cell_111/MatMul_1/ReadVariableOp%lstm_cell_111/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
lstm_107_while_cond_23133909.
*lstm_107_while_lstm_107_while_loop_counter4
0lstm_107_while_lstm_107_while_maximum_iterations
lstm_107_while_placeholder 
lstm_107_while_placeholder_1 
lstm_107_while_placeholder_2 
lstm_107_while_placeholder_30
,lstm_107_while_less_lstm_107_strided_slice_1H
Dlstm_107_while_lstm_107_while_cond_23133909___redundant_placeholder0H
Dlstm_107_while_lstm_107_while_cond_23133909___redundant_placeholder1H
Dlstm_107_while_lstm_107_while_cond_23133909___redundant_placeholder2H
Dlstm_107_while_lstm_107_while_cond_23133909___redundant_placeholder3
lstm_107_while_identity
�
lstm_107/while/LessLesslstm_107_while_placeholder,lstm_107_while_less_lstm_107_strided_slice_1*
T0*
_output_shapes
: ]
lstm_107/while/IdentityIdentitylstm_107/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_107_while_identity lstm_107/while/Identity:output:0*(
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
+__inference_lstm_107_layer_call_fn_23135252
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23131643o
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
�
�
while_cond_23134540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23134540___redundant_placeholder06
2while_while_cond_23134540___redundant_placeholder16
2while_while_cond_23134540___redundant_placeholder26
2while_while_cond_23134540___redundant_placeholder3
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
�
�
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133026
lstm_105_input#
lstm_105_23132998:x#
lstm_105_23133000:x
lstm_105_23133002:x#
lstm_106_23133005:x#
lstm_106_23133007:x
lstm_106_23133009:x#
lstm_107_23133012:x#
lstm_107_23133014:x
lstm_107_23133016:x#
dense_85_23133020:
dense_85_23133022:
identity�� dense_85/StatefulPartitionedCall� lstm_105/StatefulPartitionedCall� lstm_106/StatefulPartitionedCall� lstm_107/StatefulPartitionedCall�
 lstm_105/StatefulPartitionedCallStatefulPartitionedCalllstm_105_inputlstm_105_23132998lstm_105_23133000lstm_105_23133002*
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23131994�
 lstm_106/StatefulPartitionedCallStatefulPartitionedCall)lstm_105/StatefulPartitionedCall:output:0lstm_106_23133005lstm_106_23133007lstm_106_23133009*
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132144�
 lstm_107/StatefulPartitionedCallStatefulPartitionedCall)lstm_106/StatefulPartitionedCall:output:0lstm_107_23133012lstm_107_23133014lstm_107_23133016*
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132296�
dropout_68/PartitionedCallPartitionedCall)lstm_107/StatefulPartitionedCall:output:0*
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132309�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall#dropout_68/PartitionedCall:output:0dense_85_23133020dense_85_23133022*
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
F__inference_dense_85_layer_call_and_return_conditional_losses_23132321x
IdentityIdentity)dense_85/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_85/StatefulPartitionedCall!^lstm_105/StatefulPartitionedCall!^lstm_106/StatefulPartitionedCall!^lstm_107/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 lstm_105/StatefulPartitionedCall lstm_105/StatefulPartitionedCall2D
 lstm_106/StatefulPartitionedCall lstm_106/StatefulPartitionedCall2D
 lstm_107/StatefulPartitionedCall lstm_107/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_105_input
�
�
while_cond_23135489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23135489___redundant_placeholder06
2while_while_cond_23135489___redundant_placeholder16
2while_while_cond_23135489___redundant_placeholder26
2while_while_cond_23135489___redundant_placeholder3
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
while_body_23135157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_111_matmul_readvariableop_resource_0:xH
6while_lstm_cell_111_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_111_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_111_matmul_readvariableop_resource:xF
4while_lstm_cell_111_matmul_1_readvariableop_resource:xA
3while_lstm_cell_111_biasadd_readvariableop_resource:x��*while/lstm_cell_111/BiasAdd/ReadVariableOp�)while/lstm_cell_111/MatMul/ReadVariableOp�+while/lstm_cell_111/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_111/addAddV2$while/lstm_cell_111/MatMul:product:0&while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_111/BiasAddBiasAddwhile/lstm_cell_111/add:z:02while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_111/splitSplit,while/lstm_cell_111/split/split_dim:output:0$while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_111/SigmoidSigmoid"while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_1Sigmoid"while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mulMul!while/lstm_cell_111/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_111/ReluRelu"while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_1Mulwhile/lstm_cell_111/Sigmoid:y:0&while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/add_1AddV2while/lstm_cell_111/mul:z:0while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_2Sigmoid"while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_111/Relu_1Reluwhile/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_2Mul!while/lstm_cell_111/Sigmoid_2:y:0(while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_111/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_111/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_111/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_111/BiasAdd/ReadVariableOp*^while/lstm_cell_111/MatMul/ReadVariableOp,^while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_111_biasadd_readvariableop_resource5while_lstm_cell_111_biasadd_readvariableop_resource_0"n
4while_lstm_cell_111_matmul_1_readvariableop_resource6while_lstm_cell_111_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_111_matmul_readvariableop_resource4while_lstm_cell_111_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_111/BiasAdd/ReadVariableOp*while/lstm_cell_111/BiasAdd/ReadVariableOp2V
)while/lstm_cell_111/MatMul/ReadVariableOp)while/lstm_cell_111/MatMul/ReadVariableOp2Z
+while/lstm_cell_111/MatMul_1/ReadVariableOp+while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23130858

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
�
�
while_cond_23134727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23134727___redundant_placeholder06
2while_while_cond_23134727___redundant_placeholder16
2while_while_cond_23134727___redundant_placeholder26
2while_while_cond_23134727___redundant_placeholder3
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
while_body_23131766
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_112_23131790_0:x0
while_lstm_cell_112_23131792_0:x,
while_lstm_cell_112_23131794_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_112_23131790:x.
while_lstm_cell_112_23131792:x*
while_lstm_cell_112_23131794:x��+while/lstm_cell_112/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_112/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_112_23131790_0while_lstm_cell_112_23131792_0while_lstm_cell_112_23131794_0*
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131706r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_112/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_112/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_112/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_112/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_112_23131790while_lstm_cell_112_23131790_0">
while_lstm_cell_112_23131792while_lstm_cell_112_23131792_0">
while_lstm_cell_112_23131794while_lstm_cell_112_23131794_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_112/StatefulPartitionedCall+while/lstm_cell_112/StatefulPartitionedCall: 
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
׃
�
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133572

inputsG
5lstm_105_lstm_cell_110_matmul_readvariableop_resource:xI
7lstm_105_lstm_cell_110_matmul_1_readvariableop_resource:xD
6lstm_105_lstm_cell_110_biasadd_readvariableop_resource:xG
5lstm_106_lstm_cell_111_matmul_readvariableop_resource:xI
7lstm_106_lstm_cell_111_matmul_1_readvariableop_resource:xD
6lstm_106_lstm_cell_111_biasadd_readvariableop_resource:xG
5lstm_107_lstm_cell_112_matmul_readvariableop_resource:xI
7lstm_107_lstm_cell_112_matmul_1_readvariableop_resource:xD
6lstm_107_lstm_cell_112_biasadd_readvariableop_resource:x9
'dense_85_matmul_readvariableop_resource:6
(dense_85_biasadd_readvariableop_resource:
identity��dense_85/BiasAdd/ReadVariableOp�dense_85/MatMul/ReadVariableOp�-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp�,lstm_105/lstm_cell_110/MatMul/ReadVariableOp�.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp�lstm_105/while�-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp�,lstm_106/lstm_cell_111/MatMul/ReadVariableOp�.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp�lstm_106/while�-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp�,lstm_107/lstm_cell_112/MatMul/ReadVariableOp�.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp�lstm_107/whileD
lstm_105/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_105/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_105/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_105/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_sliceStridedSlicelstm_105/Shape:output:0%lstm_105/strided_slice/stack:output:0'lstm_105/strided_slice/stack_1:output:0'lstm_105/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_105/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/zeros/packedPacklstm_105/strided_slice:output:0 lstm_105/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_105/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_105/zerosFilllstm_105/zeros/packed:output:0lstm_105/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_105/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/zeros_1/packedPacklstm_105/strided_slice:output:0"lstm_105/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_105/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_105/zeros_1Fill lstm_105/zeros_1/packed:output:0lstm_105/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_105/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_105/transpose	Transposeinputs lstm_105/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_105/Shape_1Shapelstm_105/transpose:y:0*
T0*
_output_shapes
:h
lstm_105/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_105/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_105/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_slice_1StridedSlicelstm_105/Shape_1:output:0'lstm_105/strided_slice_1/stack:output:0)lstm_105/strided_slice_1/stack_1:output:0)lstm_105/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_105/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_105/TensorArrayV2TensorListReserve-lstm_105/TensorArrayV2/element_shape:output:0!lstm_105/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_105/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_105/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_105/transpose:y:0Glstm_105/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_105/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_105/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_105/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_slice_2StridedSlicelstm_105/transpose:y:0'lstm_105/strided_slice_2/stack:output:0)lstm_105/strided_slice_2/stack_1:output:0)lstm_105/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_105/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp5lstm_105_lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_105/lstm_cell_110/MatMulMatMul!lstm_105/strided_slice_2:output:04lstm_105/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp7lstm_105_lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_105/lstm_cell_110/MatMul_1MatMullstm_105/zeros:output:06lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_105/lstm_cell_110/addAddV2'lstm_105/lstm_cell_110/MatMul:product:0)lstm_105/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp6lstm_105_lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_105/lstm_cell_110/BiasAddBiasAddlstm_105/lstm_cell_110/add:z:05lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_105/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/lstm_cell_110/splitSplit/lstm_105/lstm_cell_110/split/split_dim:output:0'lstm_105/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_105/lstm_cell_110/SigmoidSigmoid%lstm_105/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:����������
 lstm_105/lstm_cell_110/Sigmoid_1Sigmoid%lstm_105/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/mulMul$lstm_105/lstm_cell_110/Sigmoid_1:y:0lstm_105/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_105/lstm_cell_110/ReluRelu%lstm_105/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/mul_1Mul"lstm_105/lstm_cell_110/Sigmoid:y:0)lstm_105/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/add_1AddV2lstm_105/lstm_cell_110/mul:z:0 lstm_105/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_105/lstm_cell_110/Sigmoid_2Sigmoid%lstm_105/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������y
lstm_105/lstm_cell_110/Relu_1Relu lstm_105/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/mul_2Mul$lstm_105/lstm_cell_110/Sigmoid_2:y:0+lstm_105/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_105/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_105/TensorArrayV2_1TensorListReserve/lstm_105/TensorArrayV2_1/element_shape:output:0!lstm_105/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_105/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_105/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_105/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_105/whileWhile$lstm_105/while/loop_counter:output:0*lstm_105/while/maximum_iterations:output:0lstm_105/time:output:0!lstm_105/TensorArrayV2_1:handle:0lstm_105/zeros:output:0lstm_105/zeros_1:output:0!lstm_105/strided_slice_1:output:0@lstm_105/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_105_lstm_cell_110_matmul_readvariableop_resource7lstm_105_lstm_cell_110_matmul_1_readvariableop_resource6lstm_105_lstm_cell_110_biasadd_readvariableop_resource*
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
lstm_105_while_body_23133201*(
cond R
lstm_105_while_cond_23133200*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_105/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_105/TensorArrayV2Stack/TensorListStackTensorListStacklstm_105/while:output:3Blstm_105/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_105/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_105/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_105/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_slice_3StridedSlice4lstm_105/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_105/strided_slice_3/stack:output:0)lstm_105/strided_slice_3/stack_1:output:0)lstm_105/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_105/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_105/transpose_1	Transpose4lstm_105/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_105/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_105/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_106/ShapeShapelstm_105/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_106/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_106/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_106/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_sliceStridedSlicelstm_106/Shape:output:0%lstm_106/strided_slice/stack:output:0'lstm_106/strided_slice/stack_1:output:0'lstm_106/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_106/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/zeros/packedPacklstm_106/strided_slice:output:0 lstm_106/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_106/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_106/zerosFilllstm_106/zeros/packed:output:0lstm_106/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_106/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/zeros_1/packedPacklstm_106/strided_slice:output:0"lstm_106/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_106/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_106/zeros_1Fill lstm_106/zeros_1/packed:output:0lstm_106/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_106/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_106/transpose	Transposelstm_105/transpose_1:y:0 lstm_106/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_106/Shape_1Shapelstm_106/transpose:y:0*
T0*
_output_shapes
:h
lstm_106/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_106/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_106/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_slice_1StridedSlicelstm_106/Shape_1:output:0'lstm_106/strided_slice_1/stack:output:0)lstm_106/strided_slice_1/stack_1:output:0)lstm_106/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_106/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_106/TensorArrayV2TensorListReserve-lstm_106/TensorArrayV2/element_shape:output:0!lstm_106/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_106/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_106/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_106/transpose:y:0Glstm_106/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_106/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_106/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_106/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_slice_2StridedSlicelstm_106/transpose:y:0'lstm_106/strided_slice_2/stack:output:0)lstm_106/strided_slice_2/stack_1:output:0)lstm_106/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_106/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp5lstm_106_lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_106/lstm_cell_111/MatMulMatMul!lstm_106/strided_slice_2:output:04lstm_106/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp7lstm_106_lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_106/lstm_cell_111/MatMul_1MatMullstm_106/zeros:output:06lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_106/lstm_cell_111/addAddV2'lstm_106/lstm_cell_111/MatMul:product:0)lstm_106/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp6lstm_106_lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_106/lstm_cell_111/BiasAddBiasAddlstm_106/lstm_cell_111/add:z:05lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_106/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/lstm_cell_111/splitSplit/lstm_106/lstm_cell_111/split/split_dim:output:0'lstm_106/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_106/lstm_cell_111/SigmoidSigmoid%lstm_106/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:����������
 lstm_106/lstm_cell_111/Sigmoid_1Sigmoid%lstm_106/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/mulMul$lstm_106/lstm_cell_111/Sigmoid_1:y:0lstm_106/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_106/lstm_cell_111/ReluRelu%lstm_106/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/mul_1Mul"lstm_106/lstm_cell_111/Sigmoid:y:0)lstm_106/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/add_1AddV2lstm_106/lstm_cell_111/mul:z:0 lstm_106/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_106/lstm_cell_111/Sigmoid_2Sigmoid%lstm_106/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������y
lstm_106/lstm_cell_111/Relu_1Relu lstm_106/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/mul_2Mul$lstm_106/lstm_cell_111/Sigmoid_2:y:0+lstm_106/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_106/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_106/TensorArrayV2_1TensorListReserve/lstm_106/TensorArrayV2_1/element_shape:output:0!lstm_106/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_106/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_106/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_106/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_106/whileWhile$lstm_106/while/loop_counter:output:0*lstm_106/while/maximum_iterations:output:0lstm_106/time:output:0!lstm_106/TensorArrayV2_1:handle:0lstm_106/zeros:output:0lstm_106/zeros_1:output:0!lstm_106/strided_slice_1:output:0@lstm_106/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_106_lstm_cell_111_matmul_readvariableop_resource7lstm_106_lstm_cell_111_matmul_1_readvariableop_resource6lstm_106_lstm_cell_111_biasadd_readvariableop_resource*
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
lstm_106_while_body_23133340*(
cond R
lstm_106_while_cond_23133339*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_106/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_106/TensorArrayV2Stack/TensorListStackTensorListStacklstm_106/while:output:3Blstm_106/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_106/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_106/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_106/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_slice_3StridedSlice4lstm_106/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_106/strided_slice_3/stack:output:0)lstm_106/strided_slice_3/stack_1:output:0)lstm_106/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_106/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_106/transpose_1	Transpose4lstm_106/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_106/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_106/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_107/ShapeShapelstm_106/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_107/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_107/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_107/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_sliceStridedSlicelstm_107/Shape:output:0%lstm_107/strided_slice/stack:output:0'lstm_107/strided_slice/stack_1:output:0'lstm_107/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_107/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/zeros/packedPacklstm_107/strided_slice:output:0 lstm_107/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_107/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_107/zerosFilllstm_107/zeros/packed:output:0lstm_107/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_107/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/zeros_1/packedPacklstm_107/strided_slice:output:0"lstm_107/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_107/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_107/zeros_1Fill lstm_107/zeros_1/packed:output:0lstm_107/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_107/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_107/transpose	Transposelstm_106/transpose_1:y:0 lstm_107/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_107/Shape_1Shapelstm_107/transpose:y:0*
T0*
_output_shapes
:h
lstm_107/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_107/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_107/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_slice_1StridedSlicelstm_107/Shape_1:output:0'lstm_107/strided_slice_1/stack:output:0)lstm_107/strided_slice_1/stack_1:output:0)lstm_107/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_107/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_107/TensorArrayV2TensorListReserve-lstm_107/TensorArrayV2/element_shape:output:0!lstm_107/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_107/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_107/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_107/transpose:y:0Glstm_107/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_107/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_107/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_107/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_slice_2StridedSlicelstm_107/transpose:y:0'lstm_107/strided_slice_2/stack:output:0)lstm_107/strided_slice_2/stack_1:output:0)lstm_107/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_107/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp5lstm_107_lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_107/lstm_cell_112/MatMulMatMul!lstm_107/strided_slice_2:output:04lstm_107/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp7lstm_107_lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_107/lstm_cell_112/MatMul_1MatMullstm_107/zeros:output:06lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_107/lstm_cell_112/addAddV2'lstm_107/lstm_cell_112/MatMul:product:0)lstm_107/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp6lstm_107_lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_107/lstm_cell_112/BiasAddBiasAddlstm_107/lstm_cell_112/add:z:05lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_107/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/lstm_cell_112/splitSplit/lstm_107/lstm_cell_112/split/split_dim:output:0'lstm_107/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_107/lstm_cell_112/SigmoidSigmoid%lstm_107/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:����������
 lstm_107/lstm_cell_112/Sigmoid_1Sigmoid%lstm_107/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/mulMul$lstm_107/lstm_cell_112/Sigmoid_1:y:0lstm_107/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_107/lstm_cell_112/ReluRelu%lstm_107/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/mul_1Mul"lstm_107/lstm_cell_112/Sigmoid:y:0)lstm_107/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/add_1AddV2lstm_107/lstm_cell_112/mul:z:0 lstm_107/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_107/lstm_cell_112/Sigmoid_2Sigmoid%lstm_107/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������y
lstm_107/lstm_cell_112/Relu_1Relu lstm_107/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/mul_2Mul$lstm_107/lstm_cell_112/Sigmoid_2:y:0+lstm_107/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_107/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
%lstm_107/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/TensorArrayV2_1TensorListReserve/lstm_107/TensorArrayV2_1/element_shape:output:0.lstm_107/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_107/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_107/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_107/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_107/whileWhile$lstm_107/while/loop_counter:output:0*lstm_107/while/maximum_iterations:output:0lstm_107/time:output:0!lstm_107/TensorArrayV2_1:handle:0lstm_107/zeros:output:0lstm_107/zeros_1:output:0!lstm_107/strided_slice_1:output:0@lstm_107/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_107_lstm_cell_112_matmul_readvariableop_resource7lstm_107_lstm_cell_112_matmul_1_readvariableop_resource6lstm_107_lstm_cell_112_biasadd_readvariableop_resource*
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
lstm_107_while_body_23133480*(
cond R
lstm_107_while_cond_23133479*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_107/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_107/TensorArrayV2Stack/TensorListStackTensorListStacklstm_107/while:output:3Blstm_107/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsq
lstm_107/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_107/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_107/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_slice_3StridedSlice4lstm_107/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_107/strided_slice_3/stack:output:0)lstm_107/strided_slice_3/stack_1:output:0)lstm_107/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_107/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_107/transpose_1	Transpose4lstm_107/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_107/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_107/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    t
dropout_68/IdentityIdentity!lstm_107/strided_slice_3:output:0*
T0*'
_output_shapes
:����������
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_85/MatMulMatMuldropout_68/Identity:output:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_85/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp.^lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp-^lstm_105/lstm_cell_110/MatMul/ReadVariableOp/^lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp^lstm_105/while.^lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp-^lstm_106/lstm_cell_111/MatMul/ReadVariableOp/^lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp^lstm_106/while.^lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp-^lstm_107/lstm_cell_112/MatMul/ReadVariableOp/^lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp^lstm_107/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2^
-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp2\
,lstm_105/lstm_cell_110/MatMul/ReadVariableOp,lstm_105/lstm_cell_110/MatMul/ReadVariableOp2`
.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp2 
lstm_105/whilelstm_105/while2^
-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp2\
,lstm_106/lstm_cell_111/MatMul/ReadVariableOp,lstm_106/lstm_cell_111/MatMul/ReadVariableOp2`
.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp2 
lstm_106/whilelstm_106/while2^
-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp2\
,lstm_107/lstm_cell_112/MatMul/ReadVariableOp,lstm_107/lstm_cell_112/MatMul/ReadVariableOp2`
.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp2 
lstm_107/whilelstm_107/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_23134398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_110_matmul_readvariableop_resource_0:xH
6while_lstm_cell_110_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_110_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_110_matmul_readvariableop_resource:xF
4while_lstm_cell_110_matmul_1_readvariableop_resource:xA
3while_lstm_cell_110_biasadd_readvariableop_resource:x��*while/lstm_cell_110/BiasAdd/ReadVariableOp�)while/lstm_cell_110/MatMul/ReadVariableOp�+while/lstm_cell_110/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_110/addAddV2$while/lstm_cell_110/MatMul:product:0&while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_110/BiasAddBiasAddwhile/lstm_cell_110/add:z:02while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_110/splitSplit,while/lstm_cell_110/split/split_dim:output:0$while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_110/SigmoidSigmoid"while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_1Sigmoid"while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mulMul!while/lstm_cell_110/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_110/ReluRelu"while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_1Mulwhile/lstm_cell_110/Sigmoid:y:0&while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/add_1AddV2while/lstm_cell_110/mul:z:0while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_2Sigmoid"while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_110/Relu_1Reluwhile/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_2Mul!while/lstm_cell_110/Sigmoid_2:y:0(while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_110/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_110/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_110/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_110/BiasAdd/ReadVariableOp*^while/lstm_cell_110/MatMul/ReadVariableOp,^while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_110_biasadd_readvariableop_resource5while_lstm_cell_110_biasadd_readvariableop_resource_0"n
4while_lstm_cell_110_matmul_1_readvariableop_resource6while_lstm_cell_110_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_110_matmul_readvariableop_resource4while_lstm_cell_110_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_110/BiasAdd/ReadVariableOp*while/lstm_cell_110/BiasAdd/ReadVariableOp2V
)while/lstm_cell_110/MatMul/ReadVariableOp)while/lstm_cell_110/MatMul/ReadVariableOp2Z
+while/lstm_cell_110/MatMul_1/ReadVariableOp+while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23131836

inputs(
lstm_cell_112_23131752:x(
lstm_cell_112_23131754:x$
lstm_cell_112_23131756:x
identity��%lstm_cell_112/StatefulPartitionedCall�while;
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
%lstm_cell_112/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_112_23131752lstm_cell_112_23131754lstm_cell_112_23131756*
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131706n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_112_23131752lstm_cell_112_23131754lstm_cell_112_23131756*
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
while_body_23131766*
condR
while_cond_23131765*K
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
NoOpNoOp&^lstm_cell_112/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_112/StatefulPartitionedCall%lstm_cell_112/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�K
�
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132544

inputs>
,lstm_cell_112_matmul_readvariableop_resource:x@
.lstm_cell_112_matmul_1_readvariableop_resource:x;
-lstm_cell_112_biasadd_readvariableop_resource:x
identity��$lstm_cell_112/BiasAdd/ReadVariableOp�#lstm_cell_112/MatMul/ReadVariableOp�%lstm_cell_112/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_112/MatMul/ReadVariableOpReadVariableOp,lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMulMatMulstrided_slice_2:output:0+lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMul_1MatMulzeros:output:0-lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_112/addAddV2lstm_cell_112/MatMul:product:0 lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_112/BiasAddBiasAddlstm_cell_112/add:z:0,lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_112/splitSplit&lstm_cell_112/split/split_dim:output:0lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_112/SigmoidSigmoidlstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_1Sigmoidlstm_cell_112/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_112/mulMullstm_cell_112/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_112/ReluRelulstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_1Mullstm_cell_112/Sigmoid:y:0 lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_112/add_1AddV2lstm_cell_112/mul:z:0lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_2Sigmoidlstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_112/Relu_1Relulstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_2Mullstm_cell_112/Sigmoid_2:y:0"lstm_cell_112/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_112_matmul_readvariableop_resource.lstm_cell_112_matmul_1_readvariableop_resource-lstm_cell_112_biasadd_readvariableop_resource*
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
while_body_23132459*
condR
while_cond_23132458*K
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
NoOpNoOp%^lstm_cell_112/BiasAdd/ReadVariableOp$^lstm_cell_112/MatMul/ReadVariableOp&^lstm_cell_112/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_112/BiasAdd/ReadVariableOp$lstm_cell_112/BiasAdd/ReadVariableOp2J
#lstm_cell_112/MatMul/ReadVariableOp#lstm_cell_112/MatMul/ReadVariableOp2N
%lstm_cell_112/MatMul_1/ReadVariableOp%lstm_cell_112/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
0__inference_sequential_87_layer_call_fn_23133142

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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23132943o
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
while_cond_23135156
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23135156___redundant_placeholder06
2while_while_cond_23135156___redundant_placeholder16
2while_while_cond_23135156___redundant_placeholder26
2while_while_cond_23135156___redundant_placeholder3
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
+__inference_lstm_106_layer_call_fn_23134669

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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132709s
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
�
�
while_cond_23134397
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23134397___redundant_placeholder06
2while_while_cond_23134397___redundant_placeholder16
2while_while_cond_23134397___redundant_placeholder26
2while_while_cond_23134397___redundant_placeholder3
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23130941

inputs(
lstm_cell_110_23130859:x(
lstm_cell_110_23130861:x$
lstm_cell_110_23130863:x
identity��%lstm_cell_110/StatefulPartitionedCall�while;
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
%lstm_cell_110/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_110_23130859lstm_cell_110_23130861lstm_cell_110_23130863*
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23130858n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_110_23130859lstm_cell_110_23130861lstm_cell_110_23130863*
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
while_body_23130872*
condR
while_cond_23130871*K
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
NoOpNoOp&^lstm_cell_110/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_110/StatefulPartitionedCall%lstm_cell_110/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�J
�
F__inference_lstm_105_layer_call_and_return_conditional_losses_23131994

inputs>
,lstm_cell_110_matmul_readvariableop_resource:x@
.lstm_cell_110_matmul_1_readvariableop_resource:x;
-lstm_cell_110_biasadd_readvariableop_resource:x
identity��$lstm_cell_110/BiasAdd/ReadVariableOp�#lstm_cell_110/MatMul/ReadVariableOp�%lstm_cell_110/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_110/MatMul/ReadVariableOpReadVariableOp,lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMulMatMulstrided_slice_2:output:0+lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMul_1MatMulzeros:output:0-lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_110/addAddV2lstm_cell_110/MatMul:product:0 lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_110/BiasAddBiasAddlstm_cell_110/add:z:0,lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_110/splitSplit&lstm_cell_110/split/split_dim:output:0lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_110/SigmoidSigmoidlstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_1Sigmoidlstm_cell_110/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_110/mulMullstm_cell_110/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_110/ReluRelulstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_1Mullstm_cell_110/Sigmoid:y:0 lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_110/add_1AddV2lstm_cell_110/mul:z:0lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_2Sigmoidlstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_110/Relu_1Relulstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_2Mullstm_cell_110/Sigmoid_2:y:0"lstm_cell_110/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_110_matmul_readvariableop_resource.lstm_cell_110_matmul_1_readvariableop_resource-lstm_cell_110_biasadd_readvariableop_resource*
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
while_body_23131910*
condR
while_cond_23131909*K
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
NoOpNoOp%^lstm_cell_110/BiasAdd/ReadVariableOp$^lstm_cell_110/MatMul/ReadVariableOp&^lstm_cell_110/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_110/BiasAdd/ReadVariableOp$lstm_cell_110/BiasAdd/ReadVariableOp2J
#lstm_cell_110/MatMul/ReadVariableOp#lstm_cell_110/MatMul/ReadVariableOp2N
%lstm_cell_110/MatMul_1/ReadVariableOp%lstm_cell_110/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_23132060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_111_matmul_readvariableop_resource_0:xH
6while_lstm_cell_111_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_111_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_111_matmul_readvariableop_resource:xF
4while_lstm_cell_111_matmul_1_readvariableop_resource:xA
3while_lstm_cell_111_biasadd_readvariableop_resource:x��*while/lstm_cell_111/BiasAdd/ReadVariableOp�)while/lstm_cell_111/MatMul/ReadVariableOp�+while/lstm_cell_111/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_111/addAddV2$while/lstm_cell_111/MatMul:product:0&while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_111/BiasAddBiasAddwhile/lstm_cell_111/add:z:02while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_111/splitSplit,while/lstm_cell_111/split/split_dim:output:0$while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_111/SigmoidSigmoid"while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_1Sigmoid"while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mulMul!while/lstm_cell_111/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_111/ReluRelu"while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_1Mulwhile/lstm_cell_111/Sigmoid:y:0&while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/add_1AddV2while/lstm_cell_111/mul:z:0while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_2Sigmoid"while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_111/Relu_1Reluwhile/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_2Mul!while/lstm_cell_111/Sigmoid_2:y:0(while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_111/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_111/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_111/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_111/BiasAdd/ReadVariableOp*^while/lstm_cell_111/MatMul/ReadVariableOp,^while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_111_biasadd_readvariableop_resource5while_lstm_cell_111_biasadd_readvariableop_resource_0"n
4while_lstm_cell_111_matmul_1_readvariableop_resource6while_lstm_cell_111_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_111_matmul_readvariableop_resource4while_lstm_cell_111_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_111/BiasAdd/ReadVariableOp*while/lstm_cell_111/BiasAdd/ReadVariableOp2V
)while/lstm_cell_111/MatMul/ReadVariableOp)while/lstm_cell_111/MatMul/ReadVariableOp2Z
+while/lstm_cell_111/MatMul_1/ReadVariableOp+while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
while_cond_23134870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23134870___redundant_placeholder06
2while_while_cond_23134870___redundant_placeholder16
2while_while_cond_23134870___redundant_placeholder26
2while_while_cond_23134870___redundant_placeholder3
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
while_cond_23134111
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23134111___redundant_placeholder06
2while_while_cond_23134111___redundant_placeholder16
2while_while_cond_23134111___redundant_placeholder26
2while_while_cond_23134111___redundant_placeholder3
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
while_body_23132211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_112_matmul_readvariableop_resource_0:xH
6while_lstm_cell_112_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_112_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_112_matmul_readvariableop_resource:xF
4while_lstm_cell_112_matmul_1_readvariableop_resource:xA
3while_lstm_cell_112_biasadd_readvariableop_resource:x��*while/lstm_cell_112/BiasAdd/ReadVariableOp�)while/lstm_cell_112/MatMul/ReadVariableOp�+while/lstm_cell_112/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_112/addAddV2$while/lstm_cell_112/MatMul:product:0&while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_112/BiasAddBiasAddwhile/lstm_cell_112/add:z:02while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_112/splitSplit,while/lstm_cell_112/split/split_dim:output:0$while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_112/SigmoidSigmoid"while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_1Sigmoid"while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mulMul!while/lstm_cell_112/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_112/ReluRelu"while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_1Mulwhile/lstm_cell_112/Sigmoid:y:0&while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/add_1AddV2while/lstm_cell_112/mul:z:0while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_2Sigmoid"while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_112/Relu_1Reluwhile/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_2Mul!while/lstm_cell_112/Sigmoid_2:y:0(while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_112/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_112/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_112/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_112/BiasAdd/ReadVariableOp*^while/lstm_cell_112/MatMul/ReadVariableOp,^while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_112_biasadd_readvariableop_resource5while_lstm_cell_112_biasadd_readvariableop_resource_0"n
4while_lstm_cell_112_matmul_1_readvariableop_resource6while_lstm_cell_112_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_112_matmul_readvariableop_resource4while_lstm_cell_112_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_112/BiasAdd/ReadVariableOp*while/lstm_cell_112/BiasAdd/ReadVariableOp2V
)while/lstm_cell_112/MatMul/ReadVariableOp)while/lstm_cell_112/MatMul/ReadVariableOp2Z
+while/lstm_cell_112/MatMul_1/ReadVariableOp+while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
�
�
+__inference_lstm_107_layer_call_fn_23135274

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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132296o
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
�C
�

lstm_105_while_body_23133201.
*lstm_105_while_lstm_105_while_loop_counter4
0lstm_105_while_lstm_105_while_maximum_iterations
lstm_105_while_placeholder 
lstm_105_while_placeholder_1 
lstm_105_while_placeholder_2 
lstm_105_while_placeholder_3-
)lstm_105_while_lstm_105_strided_slice_1_0i
elstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0:xQ
?lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0:xL
>lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0:x
lstm_105_while_identity
lstm_105_while_identity_1
lstm_105_while_identity_2
lstm_105_while_identity_3
lstm_105_while_identity_4
lstm_105_while_identity_5+
'lstm_105_while_lstm_105_strided_slice_1g
clstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensorM
;lstm_105_while_lstm_cell_110_matmul_readvariableop_resource:xO
=lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource:xJ
<lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource:x��3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp�2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp�4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp�
@lstm_105/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_105/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensor_0lstm_105_while_placeholderIlstm_105/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp=lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_105/while/lstm_cell_110/MatMulMatMul9lstm_105/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp?lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_105/while/lstm_cell_110/MatMul_1MatMullstm_105_while_placeholder_2<lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_105/while/lstm_cell_110/addAddV2-lstm_105/while/lstm_cell_110/MatMul:product:0/lstm_105/while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp>lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_105/while/lstm_cell_110/BiasAddBiasAdd$lstm_105/while/lstm_cell_110/add:z:0;lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_105/while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_105/while/lstm_cell_110/splitSplit5lstm_105/while/lstm_cell_110/split/split_dim:output:0-lstm_105/while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_105/while/lstm_cell_110/SigmoidSigmoid+lstm_105/while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:����������
&lstm_105/while/lstm_cell_110/Sigmoid_1Sigmoid+lstm_105/while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
 lstm_105/while/lstm_cell_110/mulMul*lstm_105/while/lstm_cell_110/Sigmoid_1:y:0lstm_105_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_105/while/lstm_cell_110/ReluRelu+lstm_105/while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
"lstm_105/while/lstm_cell_110/mul_1Mul(lstm_105/while/lstm_cell_110/Sigmoid:y:0/lstm_105/while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_105/while/lstm_cell_110/add_1AddV2$lstm_105/while/lstm_cell_110/mul:z:0&lstm_105/while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_105/while/lstm_cell_110/Sigmoid_2Sigmoid+lstm_105/while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:����������
#lstm_105/while/lstm_cell_110/Relu_1Relu&lstm_105/while/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_105/while/lstm_cell_110/mul_2Mul*lstm_105/while/lstm_cell_110/Sigmoid_2:y:01lstm_105/while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_105/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_105_while_placeholder_1lstm_105_while_placeholder&lstm_105/while/lstm_cell_110/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_105/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_105/while/addAddV2lstm_105_while_placeholderlstm_105/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_105/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/while/add_1AddV2*lstm_105_while_lstm_105_while_loop_counterlstm_105/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_105/while/IdentityIdentitylstm_105/while/add_1:z:0^lstm_105/while/NoOp*
T0*
_output_shapes
: �
lstm_105/while/Identity_1Identity0lstm_105_while_lstm_105_while_maximum_iterations^lstm_105/while/NoOp*
T0*
_output_shapes
: t
lstm_105/while/Identity_2Identitylstm_105/while/add:z:0^lstm_105/while/NoOp*
T0*
_output_shapes
: �
lstm_105/while/Identity_3IdentityClstm_105/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_105/while/NoOp*
T0*
_output_shapes
: �
lstm_105/while/Identity_4Identity&lstm_105/while/lstm_cell_110/mul_2:z:0^lstm_105/while/NoOp*
T0*'
_output_shapes
:����������
lstm_105/while/Identity_5Identity&lstm_105/while/lstm_cell_110/add_1:z:0^lstm_105/while/NoOp*
T0*'
_output_shapes
:����������
lstm_105/while/NoOpNoOp4^lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp3^lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp5^lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_105_while_identity lstm_105/while/Identity:output:0"?
lstm_105_while_identity_1"lstm_105/while/Identity_1:output:0"?
lstm_105_while_identity_2"lstm_105/while/Identity_2:output:0"?
lstm_105_while_identity_3"lstm_105/while/Identity_3:output:0"?
lstm_105_while_identity_4"lstm_105/while/Identity_4:output:0"?
lstm_105_while_identity_5"lstm_105/while/Identity_5:output:0"T
'lstm_105_while_lstm_105_strided_slice_1)lstm_105_while_lstm_105_strided_slice_1_0"~
<lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource>lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0"�
=lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource?lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0"|
;lstm_105_while_lstm_cell_110_matmul_readvariableop_resource=lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0"�
clstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensorelstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp2h
2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp2l
4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
0__inference_sequential_87_layer_call_fn_23133115

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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23132328o
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
�

g
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132383

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
�8
�
F__inference_lstm_106_layer_call_and_return_conditional_losses_23131482

inputs(
lstm_cell_111_23131400:x(
lstm_cell_111_23131402:x$
lstm_cell_111_23131404:x
identity��%lstm_cell_111/StatefulPartitionedCall�while;
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
%lstm_cell_111/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_111_23131400lstm_cell_111_23131402lstm_cell_111_23131404*
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131354n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_111_23131400lstm_cell_111_23131402lstm_cell_111_23131404*
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
while_body_23131413*
condR
while_cond_23131412*K
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
NoOpNoOp&^lstm_cell_111/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_111/StatefulPartitionedCall%lstm_cell_111/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
*sequential_87_lstm_105_while_cond_23130419J
Fsequential_87_lstm_105_while_sequential_87_lstm_105_while_loop_counterP
Lsequential_87_lstm_105_while_sequential_87_lstm_105_while_maximum_iterations,
(sequential_87_lstm_105_while_placeholder.
*sequential_87_lstm_105_while_placeholder_1.
*sequential_87_lstm_105_while_placeholder_2.
*sequential_87_lstm_105_while_placeholder_3L
Hsequential_87_lstm_105_while_less_sequential_87_lstm_105_strided_slice_1d
`sequential_87_lstm_105_while_sequential_87_lstm_105_while_cond_23130419___redundant_placeholder0d
`sequential_87_lstm_105_while_sequential_87_lstm_105_while_cond_23130419___redundant_placeholder1d
`sequential_87_lstm_105_while_sequential_87_lstm_105_while_cond_23130419___redundant_placeholder2d
`sequential_87_lstm_105_while_sequential_87_lstm_105_while_cond_23130419___redundant_placeholder3)
%sequential_87_lstm_105_while_identity
�
!sequential_87/lstm_105/while/LessLess(sequential_87_lstm_105_while_placeholderHsequential_87_lstm_105_while_less_sequential_87_lstm_105_strided_slice_1*
T0*
_output_shapes
: y
%sequential_87/lstm_105/while/IdentityIdentity%sequential_87/lstm_105/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_87_lstm_105_while_identity.sequential_87/lstm_105/while/Identity:output:0*(
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

�
lstm_107_while_cond_23133479.
*lstm_107_while_lstm_107_while_loop_counter4
0lstm_107_while_lstm_107_while_maximum_iterations
lstm_107_while_placeholder 
lstm_107_while_placeholder_1 
lstm_107_while_placeholder_2 
lstm_107_while_placeholder_30
,lstm_107_while_less_lstm_107_strided_slice_1H
Dlstm_107_while_lstm_107_while_cond_23133479___redundant_placeholder0H
Dlstm_107_while_lstm_107_while_cond_23133479___redundant_placeholder1H
Dlstm_107_while_lstm_107_while_cond_23133479___redundant_placeholder2H
Dlstm_107_while_lstm_107_while_cond_23133479___redundant_placeholder3
lstm_107_while_identity
�
lstm_107/while/LessLesslstm_107_while_placeholder,lstm_107_while_less_lstm_107_strided_slice_1*
T0*
_output_shapes
: ]
lstm_107/while/IdentityIdentitylstm_107/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_107_while_identity lstm_107/while/Identity:output:0*(
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
�
�
+__inference_lstm_107_layer_call_fn_23135285

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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132544o
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
�
�
*sequential_87_lstm_106_while_cond_23130558J
Fsequential_87_lstm_106_while_sequential_87_lstm_106_while_loop_counterP
Lsequential_87_lstm_106_while_sequential_87_lstm_106_while_maximum_iterations,
(sequential_87_lstm_106_while_placeholder.
*sequential_87_lstm_106_while_placeholder_1.
*sequential_87_lstm_106_while_placeholder_2.
*sequential_87_lstm_106_while_placeholder_3L
Hsequential_87_lstm_106_while_less_sequential_87_lstm_106_strided_slice_1d
`sequential_87_lstm_106_while_sequential_87_lstm_106_while_cond_23130558___redundant_placeholder0d
`sequential_87_lstm_106_while_sequential_87_lstm_106_while_cond_23130558___redundant_placeholder1d
`sequential_87_lstm_106_while_sequential_87_lstm_106_while_cond_23130558___redundant_placeholder2d
`sequential_87_lstm_106_while_sequential_87_lstm_106_while_cond_23130558___redundant_placeholder3)
%sequential_87_lstm_106_while_identity
�
!sequential_87/lstm_106/while/LessLess(sequential_87_lstm_106_while_placeholderHsequential_87_lstm_106_while_less_sequential_87_lstm_106_strided_slice_1*
T0*
_output_shapes
: y
%sequential_87/lstm_106/while/IdentityIdentity%sequential_87/lstm_106/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_87_lstm_106_while_identity.sequential_87/lstm_106/while/Identity:output:0*(
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
�D
�

lstm_107_while_body_23133910.
*lstm_107_while_lstm_107_while_loop_counter4
0lstm_107_while_lstm_107_while_maximum_iterations
lstm_107_while_placeholder 
lstm_107_while_placeholder_1 
lstm_107_while_placeholder_2 
lstm_107_while_placeholder_3-
)lstm_107_while_lstm_107_strided_slice_1_0i
elstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0:xQ
?lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0:xL
>lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0:x
lstm_107_while_identity
lstm_107_while_identity_1
lstm_107_while_identity_2
lstm_107_while_identity_3
lstm_107_while_identity_4
lstm_107_while_identity_5+
'lstm_107_while_lstm_107_strided_slice_1g
clstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensorM
;lstm_107_while_lstm_cell_112_matmul_readvariableop_resource:xO
=lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource:xJ
<lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource:x��3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp�2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp�4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp�
@lstm_107/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_107/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensor_0lstm_107_while_placeholderIlstm_107/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp=lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_107/while/lstm_cell_112/MatMulMatMul9lstm_107/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp?lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_107/while/lstm_cell_112/MatMul_1MatMullstm_107_while_placeholder_2<lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_107/while/lstm_cell_112/addAddV2-lstm_107/while/lstm_cell_112/MatMul:product:0/lstm_107/while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp>lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_107/while/lstm_cell_112/BiasAddBiasAdd$lstm_107/while/lstm_cell_112/add:z:0;lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_107/while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_107/while/lstm_cell_112/splitSplit5lstm_107/while/lstm_cell_112/split/split_dim:output:0-lstm_107/while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_107/while/lstm_cell_112/SigmoidSigmoid+lstm_107/while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:����������
&lstm_107/while/lstm_cell_112/Sigmoid_1Sigmoid+lstm_107/while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
 lstm_107/while/lstm_cell_112/mulMul*lstm_107/while/lstm_cell_112/Sigmoid_1:y:0lstm_107_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_107/while/lstm_cell_112/ReluRelu+lstm_107/while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
"lstm_107/while/lstm_cell_112/mul_1Mul(lstm_107/while/lstm_cell_112/Sigmoid:y:0/lstm_107/while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_107/while/lstm_cell_112/add_1AddV2$lstm_107/while/lstm_cell_112/mul:z:0&lstm_107/while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_107/while/lstm_cell_112/Sigmoid_2Sigmoid+lstm_107/while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:����������
#lstm_107/while/lstm_cell_112/Relu_1Relu&lstm_107/while/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_107/while/lstm_cell_112/mul_2Mul*lstm_107/while/lstm_cell_112/Sigmoid_2:y:01lstm_107/while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������{
9lstm_107/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
3lstm_107/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_107_while_placeholder_1Blstm_107/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_107/while/lstm_cell_112/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_107/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_107/while/addAddV2lstm_107_while_placeholderlstm_107/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_107/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/while/add_1AddV2*lstm_107_while_lstm_107_while_loop_counterlstm_107/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_107/while/IdentityIdentitylstm_107/while/add_1:z:0^lstm_107/while/NoOp*
T0*
_output_shapes
: �
lstm_107/while/Identity_1Identity0lstm_107_while_lstm_107_while_maximum_iterations^lstm_107/while/NoOp*
T0*
_output_shapes
: t
lstm_107/while/Identity_2Identitylstm_107/while/add:z:0^lstm_107/while/NoOp*
T0*
_output_shapes
: �
lstm_107/while/Identity_3IdentityClstm_107/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_107/while/NoOp*
T0*
_output_shapes
: �
lstm_107/while/Identity_4Identity&lstm_107/while/lstm_cell_112/mul_2:z:0^lstm_107/while/NoOp*
T0*'
_output_shapes
:����������
lstm_107/while/Identity_5Identity&lstm_107/while/lstm_cell_112/add_1:z:0^lstm_107/while/NoOp*
T0*'
_output_shapes
:����������
lstm_107/while/NoOpNoOp4^lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp3^lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp5^lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_107_while_identity lstm_107/while/Identity:output:0"?
lstm_107_while_identity_1"lstm_107/while/Identity_1:output:0"?
lstm_107_while_identity_2"lstm_107/while/Identity_2:output:0"?
lstm_107_while_identity_3"lstm_107/while/Identity_3:output:0"?
lstm_107_while_identity_4"lstm_107/while/Identity_4:output:0"?
lstm_107_while_identity_5"lstm_107/while/Identity_5:output:0"T
'lstm_107_while_lstm_107_strided_slice_1)lstm_107_while_lstm_107_strided_slice_1_0"~
<lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource>lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0"�
=lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource?lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0"|
;lstm_107_while_lstm_cell_112_matmul_readvariableop_resource=lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0"�
clstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensorelstm_107_while_tensorarrayv2read_tensorlistgetitem_lstm_107_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp3lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp2h
2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp2lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp2l
4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp4lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23131291

inputs(
lstm_cell_111_23131209:x(
lstm_cell_111_23131211:x$
lstm_cell_111_23131213:x
identity��%lstm_cell_111/StatefulPartitionedCall�while;
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
%lstm_cell_111/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_111_23131209lstm_cell_111_23131211lstm_cell_111_23131213*
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131208n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_111_23131209lstm_cell_111_23131211lstm_cell_111_23131213*
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
while_body_23131222*
condR
while_cond_23131221*K
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
NoOpNoOp&^lstm_cell_111/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_111/StatefulPartitionedCall%lstm_cell_111/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
й
�
#__inference__wrapped_model_23130791
lstm_105_inputU
Csequential_87_lstm_105_lstm_cell_110_matmul_readvariableop_resource:xW
Esequential_87_lstm_105_lstm_cell_110_matmul_1_readvariableop_resource:xR
Dsequential_87_lstm_105_lstm_cell_110_biasadd_readvariableop_resource:xU
Csequential_87_lstm_106_lstm_cell_111_matmul_readvariableop_resource:xW
Esequential_87_lstm_106_lstm_cell_111_matmul_1_readvariableop_resource:xR
Dsequential_87_lstm_106_lstm_cell_111_biasadd_readvariableop_resource:xU
Csequential_87_lstm_107_lstm_cell_112_matmul_readvariableop_resource:xW
Esequential_87_lstm_107_lstm_cell_112_matmul_1_readvariableop_resource:xR
Dsequential_87_lstm_107_lstm_cell_112_biasadd_readvariableop_resource:xG
5sequential_87_dense_85_matmul_readvariableop_resource:D
6sequential_87_dense_85_biasadd_readvariableop_resource:
identity��-sequential_87/dense_85/BiasAdd/ReadVariableOp�,sequential_87/dense_85/MatMul/ReadVariableOp�;sequential_87/lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp�:sequential_87/lstm_105/lstm_cell_110/MatMul/ReadVariableOp�<sequential_87/lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp�sequential_87/lstm_105/while�;sequential_87/lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp�:sequential_87/lstm_106/lstm_cell_111/MatMul/ReadVariableOp�<sequential_87/lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp�sequential_87/lstm_106/while�;sequential_87/lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp�:sequential_87/lstm_107/lstm_cell_112/MatMul/ReadVariableOp�<sequential_87/lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp�sequential_87/lstm_107/whileZ
sequential_87/lstm_105/ShapeShapelstm_105_input*
T0*
_output_shapes
:t
*sequential_87/lstm_105/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_87/lstm_105/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_87/lstm_105/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_87/lstm_105/strided_sliceStridedSlice%sequential_87/lstm_105/Shape:output:03sequential_87/lstm_105/strided_slice/stack:output:05sequential_87/lstm_105/strided_slice/stack_1:output:05sequential_87/lstm_105/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_87/lstm_105/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
#sequential_87/lstm_105/zeros/packedPack-sequential_87/lstm_105/strided_slice:output:0.sequential_87/lstm_105/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_87/lstm_105/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_87/lstm_105/zerosFill,sequential_87/lstm_105/zeros/packed:output:0+sequential_87/lstm_105/zeros/Const:output:0*
T0*'
_output_shapes
:���������i
'sequential_87/lstm_105/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%sequential_87/lstm_105/zeros_1/packedPack-sequential_87/lstm_105/strided_slice:output:00sequential_87/lstm_105/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_87/lstm_105/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_87/lstm_105/zeros_1Fill.sequential_87/lstm_105/zeros_1/packed:output:0-sequential_87/lstm_105/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������z
%sequential_87/lstm_105/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_87/lstm_105/transpose	Transposelstm_105_input.sequential_87/lstm_105/transpose/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_87/lstm_105/Shape_1Shape$sequential_87/lstm_105/transpose:y:0*
T0*
_output_shapes
:v
,sequential_87/lstm_105/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_105/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_87/lstm_105/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_105/strided_slice_1StridedSlice'sequential_87/lstm_105/Shape_1:output:05sequential_87/lstm_105/strided_slice_1/stack:output:07sequential_87/lstm_105/strided_slice_1/stack_1:output:07sequential_87/lstm_105/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_87/lstm_105/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_87/lstm_105/TensorArrayV2TensorListReserve;sequential_87/lstm_105/TensorArrayV2/element_shape:output:0/sequential_87/lstm_105/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lsequential_87/lstm_105/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_87/lstm_105/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_87/lstm_105/transpose:y:0Usequential_87/lstm_105/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,sequential_87/lstm_105/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_105/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_87/lstm_105/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_105/strided_slice_2StridedSlice$sequential_87/lstm_105/transpose:y:05sequential_87/lstm_105/strided_slice_2/stack:output:07sequential_87/lstm_105/strided_slice_2/stack_1:output:07sequential_87/lstm_105/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
:sequential_87/lstm_105/lstm_cell_110/MatMul/ReadVariableOpReadVariableOpCsequential_87_lstm_105_lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
+sequential_87/lstm_105/lstm_cell_110/MatMulMatMul/sequential_87/lstm_105/strided_slice_2:output:0Bsequential_87/lstm_105/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
<sequential_87/lstm_105/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOpEsequential_87_lstm_105_lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
-sequential_87/lstm_105/lstm_cell_110/MatMul_1MatMul%sequential_87/lstm_105/zeros:output:0Dsequential_87/lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
(sequential_87/lstm_105/lstm_cell_110/addAddV25sequential_87/lstm_105/lstm_cell_110/MatMul:product:07sequential_87/lstm_105/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
;sequential_87/lstm_105/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOpDsequential_87_lstm_105_lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
,sequential_87/lstm_105/lstm_cell_110/BiasAddBiasAdd,sequential_87/lstm_105/lstm_cell_110/add:z:0Csequential_87/lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xv
4sequential_87/lstm_105/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_87/lstm_105/lstm_cell_110/splitSplit=sequential_87/lstm_105/lstm_cell_110/split/split_dim:output:05sequential_87/lstm_105/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
,sequential_87/lstm_105/lstm_cell_110/SigmoidSigmoid3sequential_87/lstm_105/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:����������
.sequential_87/lstm_105/lstm_cell_110/Sigmoid_1Sigmoid3sequential_87/lstm_105/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
(sequential_87/lstm_105/lstm_cell_110/mulMul2sequential_87/lstm_105/lstm_cell_110/Sigmoid_1:y:0'sequential_87/lstm_105/zeros_1:output:0*
T0*'
_output_shapes
:����������
)sequential_87/lstm_105/lstm_cell_110/ReluRelu3sequential_87/lstm_105/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
*sequential_87/lstm_105/lstm_cell_110/mul_1Mul0sequential_87/lstm_105/lstm_cell_110/Sigmoid:y:07sequential_87/lstm_105/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
*sequential_87/lstm_105/lstm_cell_110/add_1AddV2,sequential_87/lstm_105/lstm_cell_110/mul:z:0.sequential_87/lstm_105/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:����������
.sequential_87/lstm_105/lstm_cell_110/Sigmoid_2Sigmoid3sequential_87/lstm_105/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:����������
+sequential_87/lstm_105/lstm_cell_110/Relu_1Relu.sequential_87/lstm_105/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
*sequential_87/lstm_105/lstm_cell_110/mul_2Mul2sequential_87/lstm_105/lstm_cell_110/Sigmoid_2:y:09sequential_87/lstm_105/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_105/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
&sequential_87/lstm_105/TensorArrayV2_1TensorListReserve=sequential_87/lstm_105/TensorArrayV2_1/element_shape:output:0/sequential_87/lstm_105/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
sequential_87/lstm_105/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_87/lstm_105/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������k
)sequential_87/lstm_105/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_87/lstm_105/whileWhile2sequential_87/lstm_105/while/loop_counter:output:08sequential_87/lstm_105/while/maximum_iterations:output:0$sequential_87/lstm_105/time:output:0/sequential_87/lstm_105/TensorArrayV2_1:handle:0%sequential_87/lstm_105/zeros:output:0'sequential_87/lstm_105/zeros_1:output:0/sequential_87/lstm_105/strided_slice_1:output:0Nsequential_87/lstm_105/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_87_lstm_105_lstm_cell_110_matmul_readvariableop_resourceEsequential_87_lstm_105_lstm_cell_110_matmul_1_readvariableop_resourceDsequential_87_lstm_105_lstm_cell_110_biasadd_readvariableop_resource*
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
*sequential_87_lstm_105_while_body_23130420*6
cond.R,
*sequential_87_lstm_105_while_cond_23130419*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Gsequential_87/lstm_105/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
9sequential_87/lstm_105/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_87/lstm_105/while:output:3Psequential_87/lstm_105/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0
,sequential_87/lstm_105/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.sequential_87/lstm_105/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_105/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_105/strided_slice_3StridedSliceBsequential_87/lstm_105/TensorArrayV2Stack/TensorListStack:tensor:05sequential_87/lstm_105/strided_slice_3/stack:output:07sequential_87/lstm_105/strided_slice_3/stack_1:output:07sequential_87/lstm_105/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask|
'sequential_87/lstm_105/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"sequential_87/lstm_105/transpose_1	TransposeBsequential_87/lstm_105/TensorArrayV2Stack/TensorListStack:tensor:00sequential_87/lstm_105/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_87/lstm_105/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    r
sequential_87/lstm_106/ShapeShape&sequential_87/lstm_105/transpose_1:y:0*
T0*
_output_shapes
:t
*sequential_87/lstm_106/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_87/lstm_106/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_87/lstm_106/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_87/lstm_106/strided_sliceStridedSlice%sequential_87/lstm_106/Shape:output:03sequential_87/lstm_106/strided_slice/stack:output:05sequential_87/lstm_106/strided_slice/stack_1:output:05sequential_87/lstm_106/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_87/lstm_106/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
#sequential_87/lstm_106/zeros/packedPack-sequential_87/lstm_106/strided_slice:output:0.sequential_87/lstm_106/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_87/lstm_106/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_87/lstm_106/zerosFill,sequential_87/lstm_106/zeros/packed:output:0+sequential_87/lstm_106/zeros/Const:output:0*
T0*'
_output_shapes
:���������i
'sequential_87/lstm_106/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%sequential_87/lstm_106/zeros_1/packedPack-sequential_87/lstm_106/strided_slice:output:00sequential_87/lstm_106/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_87/lstm_106/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_87/lstm_106/zeros_1Fill.sequential_87/lstm_106/zeros_1/packed:output:0-sequential_87/lstm_106/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������z
%sequential_87/lstm_106/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_87/lstm_106/transpose	Transpose&sequential_87/lstm_105/transpose_1:y:0.sequential_87/lstm_106/transpose/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_87/lstm_106/Shape_1Shape$sequential_87/lstm_106/transpose:y:0*
T0*
_output_shapes
:v
,sequential_87/lstm_106/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_106/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_87/lstm_106/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_106/strided_slice_1StridedSlice'sequential_87/lstm_106/Shape_1:output:05sequential_87/lstm_106/strided_slice_1/stack:output:07sequential_87/lstm_106/strided_slice_1/stack_1:output:07sequential_87/lstm_106/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_87/lstm_106/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_87/lstm_106/TensorArrayV2TensorListReserve;sequential_87/lstm_106/TensorArrayV2/element_shape:output:0/sequential_87/lstm_106/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lsequential_87/lstm_106/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_87/lstm_106/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_87/lstm_106/transpose:y:0Usequential_87/lstm_106/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,sequential_87/lstm_106/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_106/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_87/lstm_106/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_106/strided_slice_2StridedSlice$sequential_87/lstm_106/transpose:y:05sequential_87/lstm_106/strided_slice_2/stack:output:07sequential_87/lstm_106/strided_slice_2/stack_1:output:07sequential_87/lstm_106/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
:sequential_87/lstm_106/lstm_cell_111/MatMul/ReadVariableOpReadVariableOpCsequential_87_lstm_106_lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
+sequential_87/lstm_106/lstm_cell_111/MatMulMatMul/sequential_87/lstm_106/strided_slice_2:output:0Bsequential_87/lstm_106/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
<sequential_87/lstm_106/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOpEsequential_87_lstm_106_lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
-sequential_87/lstm_106/lstm_cell_111/MatMul_1MatMul%sequential_87/lstm_106/zeros:output:0Dsequential_87/lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
(sequential_87/lstm_106/lstm_cell_111/addAddV25sequential_87/lstm_106/lstm_cell_111/MatMul:product:07sequential_87/lstm_106/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
;sequential_87/lstm_106/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOpDsequential_87_lstm_106_lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
,sequential_87/lstm_106/lstm_cell_111/BiasAddBiasAdd,sequential_87/lstm_106/lstm_cell_111/add:z:0Csequential_87/lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xv
4sequential_87/lstm_106/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_87/lstm_106/lstm_cell_111/splitSplit=sequential_87/lstm_106/lstm_cell_111/split/split_dim:output:05sequential_87/lstm_106/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
,sequential_87/lstm_106/lstm_cell_111/SigmoidSigmoid3sequential_87/lstm_106/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:����������
.sequential_87/lstm_106/lstm_cell_111/Sigmoid_1Sigmoid3sequential_87/lstm_106/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
(sequential_87/lstm_106/lstm_cell_111/mulMul2sequential_87/lstm_106/lstm_cell_111/Sigmoid_1:y:0'sequential_87/lstm_106/zeros_1:output:0*
T0*'
_output_shapes
:����������
)sequential_87/lstm_106/lstm_cell_111/ReluRelu3sequential_87/lstm_106/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
*sequential_87/lstm_106/lstm_cell_111/mul_1Mul0sequential_87/lstm_106/lstm_cell_111/Sigmoid:y:07sequential_87/lstm_106/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
*sequential_87/lstm_106/lstm_cell_111/add_1AddV2,sequential_87/lstm_106/lstm_cell_111/mul:z:0.sequential_87/lstm_106/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:����������
.sequential_87/lstm_106/lstm_cell_111/Sigmoid_2Sigmoid3sequential_87/lstm_106/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:����������
+sequential_87/lstm_106/lstm_cell_111/Relu_1Relu.sequential_87/lstm_106/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
*sequential_87/lstm_106/lstm_cell_111/mul_2Mul2sequential_87/lstm_106/lstm_cell_111/Sigmoid_2:y:09sequential_87/lstm_106/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_106/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
&sequential_87/lstm_106/TensorArrayV2_1TensorListReserve=sequential_87/lstm_106/TensorArrayV2_1/element_shape:output:0/sequential_87/lstm_106/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
sequential_87/lstm_106/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_87/lstm_106/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������k
)sequential_87/lstm_106/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_87/lstm_106/whileWhile2sequential_87/lstm_106/while/loop_counter:output:08sequential_87/lstm_106/while/maximum_iterations:output:0$sequential_87/lstm_106/time:output:0/sequential_87/lstm_106/TensorArrayV2_1:handle:0%sequential_87/lstm_106/zeros:output:0'sequential_87/lstm_106/zeros_1:output:0/sequential_87/lstm_106/strided_slice_1:output:0Nsequential_87/lstm_106/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_87_lstm_106_lstm_cell_111_matmul_readvariableop_resourceEsequential_87_lstm_106_lstm_cell_111_matmul_1_readvariableop_resourceDsequential_87_lstm_106_lstm_cell_111_biasadd_readvariableop_resource*
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
*sequential_87_lstm_106_while_body_23130559*6
cond.R,
*sequential_87_lstm_106_while_cond_23130558*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Gsequential_87/lstm_106/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
9sequential_87/lstm_106/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_87/lstm_106/while:output:3Psequential_87/lstm_106/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0
,sequential_87/lstm_106/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.sequential_87/lstm_106/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_106/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_106/strided_slice_3StridedSliceBsequential_87/lstm_106/TensorArrayV2Stack/TensorListStack:tensor:05sequential_87/lstm_106/strided_slice_3/stack:output:07sequential_87/lstm_106/strided_slice_3/stack_1:output:07sequential_87/lstm_106/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask|
'sequential_87/lstm_106/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"sequential_87/lstm_106/transpose_1	TransposeBsequential_87/lstm_106/TensorArrayV2Stack/TensorListStack:tensor:00sequential_87/lstm_106/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_87/lstm_106/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    r
sequential_87/lstm_107/ShapeShape&sequential_87/lstm_106/transpose_1:y:0*
T0*
_output_shapes
:t
*sequential_87/lstm_107/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_87/lstm_107/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_87/lstm_107/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$sequential_87/lstm_107/strided_sliceStridedSlice%sequential_87/lstm_107/Shape:output:03sequential_87/lstm_107/strided_slice/stack:output:05sequential_87/lstm_107/strided_slice/stack_1:output:05sequential_87/lstm_107/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_87/lstm_107/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
#sequential_87/lstm_107/zeros/packedPack-sequential_87/lstm_107/strided_slice:output:0.sequential_87/lstm_107/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_87/lstm_107/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_87/lstm_107/zerosFill,sequential_87/lstm_107/zeros/packed:output:0+sequential_87/lstm_107/zeros/Const:output:0*
T0*'
_output_shapes
:���������i
'sequential_87/lstm_107/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%sequential_87/lstm_107/zeros_1/packedPack-sequential_87/lstm_107/strided_slice:output:00sequential_87/lstm_107/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_87/lstm_107/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_87/lstm_107/zeros_1Fill.sequential_87/lstm_107/zeros_1/packed:output:0-sequential_87/lstm_107/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������z
%sequential_87/lstm_107/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
 sequential_87/lstm_107/transpose	Transpose&sequential_87/lstm_106/transpose_1:y:0.sequential_87/lstm_107/transpose/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_87/lstm_107/Shape_1Shape$sequential_87/lstm_107/transpose:y:0*
T0*
_output_shapes
:v
,sequential_87/lstm_107/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_107/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_87/lstm_107/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_107/strided_slice_1StridedSlice'sequential_87/lstm_107/Shape_1:output:05sequential_87/lstm_107/strided_slice_1/stack:output:07sequential_87/lstm_107/strided_slice_1/stack_1:output:07sequential_87/lstm_107/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_87/lstm_107/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_87/lstm_107/TensorArrayV2TensorListReserve;sequential_87/lstm_107/TensorArrayV2/element_shape:output:0/sequential_87/lstm_107/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Lsequential_87/lstm_107/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
>sequential_87/lstm_107/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_87/lstm_107/transpose:y:0Usequential_87/lstm_107/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���v
,sequential_87/lstm_107/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_107/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_87/lstm_107/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_107/strided_slice_2StridedSlice$sequential_87/lstm_107/transpose:y:05sequential_87/lstm_107/strided_slice_2/stack:output:07sequential_87/lstm_107/strided_slice_2/stack_1:output:07sequential_87/lstm_107/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
:sequential_87/lstm_107/lstm_cell_112/MatMul/ReadVariableOpReadVariableOpCsequential_87_lstm_107_lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
+sequential_87/lstm_107/lstm_cell_112/MatMulMatMul/sequential_87/lstm_107/strided_slice_2:output:0Bsequential_87/lstm_107/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
<sequential_87/lstm_107/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOpEsequential_87_lstm_107_lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
-sequential_87/lstm_107/lstm_cell_112/MatMul_1MatMul%sequential_87/lstm_107/zeros:output:0Dsequential_87/lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
(sequential_87/lstm_107/lstm_cell_112/addAddV25sequential_87/lstm_107/lstm_cell_112/MatMul:product:07sequential_87/lstm_107/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
;sequential_87/lstm_107/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOpDsequential_87_lstm_107_lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
,sequential_87/lstm_107/lstm_cell_112/BiasAddBiasAdd,sequential_87/lstm_107/lstm_cell_112/add:z:0Csequential_87/lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xv
4sequential_87/lstm_107/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
*sequential_87/lstm_107/lstm_cell_112/splitSplit=sequential_87/lstm_107/lstm_cell_112/split/split_dim:output:05sequential_87/lstm_107/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
,sequential_87/lstm_107/lstm_cell_112/SigmoidSigmoid3sequential_87/lstm_107/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:����������
.sequential_87/lstm_107/lstm_cell_112/Sigmoid_1Sigmoid3sequential_87/lstm_107/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
(sequential_87/lstm_107/lstm_cell_112/mulMul2sequential_87/lstm_107/lstm_cell_112/Sigmoid_1:y:0'sequential_87/lstm_107/zeros_1:output:0*
T0*'
_output_shapes
:����������
)sequential_87/lstm_107/lstm_cell_112/ReluRelu3sequential_87/lstm_107/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
*sequential_87/lstm_107/lstm_cell_112/mul_1Mul0sequential_87/lstm_107/lstm_cell_112/Sigmoid:y:07sequential_87/lstm_107/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
*sequential_87/lstm_107/lstm_cell_112/add_1AddV2,sequential_87/lstm_107/lstm_cell_112/mul:z:0.sequential_87/lstm_107/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:����������
.sequential_87/lstm_107/lstm_cell_112/Sigmoid_2Sigmoid3sequential_87/lstm_107/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:����������
+sequential_87/lstm_107/lstm_cell_112/Relu_1Relu.sequential_87/lstm_107/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
*sequential_87/lstm_107/lstm_cell_112/mul_2Mul2sequential_87/lstm_107/lstm_cell_112/Sigmoid_2:y:09sequential_87/lstm_107/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_107/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   u
3sequential_87/lstm_107/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
&sequential_87/lstm_107/TensorArrayV2_1TensorListReserve=sequential_87/lstm_107/TensorArrayV2_1/element_shape:output:0<sequential_87/lstm_107/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���]
sequential_87/lstm_107/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_87/lstm_107/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������k
)sequential_87/lstm_107/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
sequential_87/lstm_107/whileWhile2sequential_87/lstm_107/while/loop_counter:output:08sequential_87/lstm_107/while/maximum_iterations:output:0$sequential_87/lstm_107/time:output:0/sequential_87/lstm_107/TensorArrayV2_1:handle:0%sequential_87/lstm_107/zeros:output:0'sequential_87/lstm_107/zeros_1:output:0/sequential_87/lstm_107/strided_slice_1:output:0Nsequential_87/lstm_107/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_87_lstm_107_lstm_cell_112_matmul_readvariableop_resourceEsequential_87_lstm_107_lstm_cell_112_matmul_1_readvariableop_resourceDsequential_87_lstm_107_lstm_cell_112_biasadd_readvariableop_resource*
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
*sequential_87_lstm_107_while_body_23130699*6
cond.R,
*sequential_87_lstm_107_while_cond_23130698*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Gsequential_87/lstm_107/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
9sequential_87/lstm_107/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_87/lstm_107/while:output:3Psequential_87/lstm_107/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elements
,sequential_87/lstm_107/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������x
.sequential_87/lstm_107/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_87/lstm_107/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&sequential_87/lstm_107/strided_slice_3StridedSliceBsequential_87/lstm_107/TensorArrayV2Stack/TensorListStack:tensor:05sequential_87/lstm_107/strided_slice_3/stack:output:07sequential_87/lstm_107/strided_slice_3/stack_1:output:07sequential_87/lstm_107/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask|
'sequential_87/lstm_107/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
"sequential_87/lstm_107/transpose_1	TransposeBsequential_87/lstm_107/TensorArrayV2Stack/TensorListStack:tensor:00sequential_87/lstm_107/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������r
sequential_87/lstm_107/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
!sequential_87/dropout_68/IdentityIdentity/sequential_87/lstm_107/strided_slice_3:output:0*
T0*'
_output_shapes
:����������
,sequential_87/dense_85/MatMul/ReadVariableOpReadVariableOp5sequential_87_dense_85_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_87/dense_85/MatMulMatMul*sequential_87/dropout_68/Identity:output:04sequential_87/dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_87/dense_85/BiasAdd/ReadVariableOpReadVariableOp6sequential_87_dense_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_87/dense_85/BiasAddBiasAdd'sequential_87/dense_85/MatMul:product:05sequential_87/dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_87/dense_85/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_87/dense_85/BiasAdd/ReadVariableOp-^sequential_87/dense_85/MatMul/ReadVariableOp<^sequential_87/lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp;^sequential_87/lstm_105/lstm_cell_110/MatMul/ReadVariableOp=^sequential_87/lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp^sequential_87/lstm_105/while<^sequential_87/lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp;^sequential_87/lstm_106/lstm_cell_111/MatMul/ReadVariableOp=^sequential_87/lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp^sequential_87/lstm_106/while<^sequential_87/lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp;^sequential_87/lstm_107/lstm_cell_112/MatMul/ReadVariableOp=^sequential_87/lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp^sequential_87/lstm_107/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2^
-sequential_87/dense_85/BiasAdd/ReadVariableOp-sequential_87/dense_85/BiasAdd/ReadVariableOp2\
,sequential_87/dense_85/MatMul/ReadVariableOp,sequential_87/dense_85/MatMul/ReadVariableOp2z
;sequential_87/lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp;sequential_87/lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp2x
:sequential_87/lstm_105/lstm_cell_110/MatMul/ReadVariableOp:sequential_87/lstm_105/lstm_cell_110/MatMul/ReadVariableOp2|
<sequential_87/lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp<sequential_87/lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp2<
sequential_87/lstm_105/whilesequential_87/lstm_105/while2z
;sequential_87/lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp;sequential_87/lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp2x
:sequential_87/lstm_106/lstm_cell_111/MatMul/ReadVariableOp:sequential_87/lstm_106/lstm_cell_111/MatMul/ReadVariableOp2|
<sequential_87/lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp<sequential_87/lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp2<
sequential_87/lstm_106/whilesequential_87/lstm_106/while2z
;sequential_87/lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp;sequential_87/lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp2x
:sequential_87/lstm_107/lstm_cell_112/MatMul/ReadVariableOp:sequential_87/lstm_107/lstm_cell_112/MatMul/ReadVariableOp2|
<sequential_87/lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp<sequential_87/lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp2<
sequential_87/lstm_107/whilesequential_87/lstm_107/while:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_105_input
�#
�
while_body_23131063
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_110_23131087_0:x0
while_lstm_cell_110_23131089_0:x,
while_lstm_cell_110_23131091_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_110_23131087:x.
while_lstm_cell_110_23131089:x*
while_lstm_cell_110_23131091:x��+while/lstm_cell_110/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_110/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_110_23131087_0while_lstm_cell_110_23131089_0while_lstm_cell_110_23131091_0*
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23131004�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_110/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_110/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_110/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_110/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_110_23131087while_lstm_cell_110_23131087_0">
while_lstm_cell_110_23131089while_lstm_cell_110_23131089_0">
while_lstm_cell_110_23131091while_lstm_cell_110_23131091_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_110/StatefulPartitionedCall+while/lstm_cell_110/StatefulPartitionedCall: 
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
while_cond_23135344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23135344___redundant_placeholder06
2while_while_cond_23135344___redundant_placeholder16
2while_while_cond_23135344___redundant_placeholder26
2while_while_cond_23135344___redundant_placeholder3
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134482

inputs>
,lstm_cell_110_matmul_readvariableop_resource:x@
.lstm_cell_110_matmul_1_readvariableop_resource:x;
-lstm_cell_110_biasadd_readvariableop_resource:x
identity��$lstm_cell_110/BiasAdd/ReadVariableOp�#lstm_cell_110/MatMul/ReadVariableOp�%lstm_cell_110/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_110/MatMul/ReadVariableOpReadVariableOp,lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMulMatMulstrided_slice_2:output:0+lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMul_1MatMulzeros:output:0-lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_110/addAddV2lstm_cell_110/MatMul:product:0 lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_110/BiasAddBiasAddlstm_cell_110/add:z:0,lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_110/splitSplit&lstm_cell_110/split/split_dim:output:0lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_110/SigmoidSigmoidlstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_1Sigmoidlstm_cell_110/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_110/mulMullstm_cell_110/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_110/ReluRelulstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_1Mullstm_cell_110/Sigmoid:y:0 lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_110/add_1AddV2lstm_cell_110/mul:z:0lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_2Sigmoidlstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_110/Relu_1Relulstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_2Mullstm_cell_110/Sigmoid_2:y:0"lstm_cell_110/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_110_matmul_readvariableop_resource.lstm_cell_110_matmul_1_readvariableop_resource-lstm_cell_110_biasadd_readvariableop_resource*
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
while_body_23134398*
condR
while_cond_23134397*K
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
NoOpNoOp%^lstm_cell_110/BiasAdd/ReadVariableOp$^lstm_cell_110/MatMul/ReadVariableOp&^lstm_cell_110/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_110/BiasAdd/ReadVariableOp$lstm_cell_110/BiasAdd/ReadVariableOp2J
#lstm_cell_110/MatMul/ReadVariableOp#lstm_cell_110/MatMul/ReadVariableOp2N
%lstm_cell_110/MatMul_1/ReadVariableOp%lstm_cell_110/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_lstm_105_layer_call_fn_23134053

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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23132874s
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
�
+__inference_lstm_106_layer_call_fn_23134647
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23131482|
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
ŋ
�
K__inference_sequential_87_layer_call_and_return_conditional_losses_23134009

inputsG
5lstm_105_lstm_cell_110_matmul_readvariableop_resource:xI
7lstm_105_lstm_cell_110_matmul_1_readvariableop_resource:xD
6lstm_105_lstm_cell_110_biasadd_readvariableop_resource:xG
5lstm_106_lstm_cell_111_matmul_readvariableop_resource:xI
7lstm_106_lstm_cell_111_matmul_1_readvariableop_resource:xD
6lstm_106_lstm_cell_111_biasadd_readvariableop_resource:xG
5lstm_107_lstm_cell_112_matmul_readvariableop_resource:xI
7lstm_107_lstm_cell_112_matmul_1_readvariableop_resource:xD
6lstm_107_lstm_cell_112_biasadd_readvariableop_resource:x9
'dense_85_matmul_readvariableop_resource:6
(dense_85_biasadd_readvariableop_resource:
identity��dense_85/BiasAdd/ReadVariableOp�dense_85/MatMul/ReadVariableOp�-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp�,lstm_105/lstm_cell_110/MatMul/ReadVariableOp�.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp�lstm_105/while�-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp�,lstm_106/lstm_cell_111/MatMul/ReadVariableOp�.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp�lstm_106/while�-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp�,lstm_107/lstm_cell_112/MatMul/ReadVariableOp�.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp�lstm_107/whileD
lstm_105/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_105/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_105/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_105/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_sliceStridedSlicelstm_105/Shape:output:0%lstm_105/strided_slice/stack:output:0'lstm_105/strided_slice/stack_1:output:0'lstm_105/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_105/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/zeros/packedPacklstm_105/strided_slice:output:0 lstm_105/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_105/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_105/zerosFilllstm_105/zeros/packed:output:0lstm_105/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_105/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/zeros_1/packedPacklstm_105/strided_slice:output:0"lstm_105/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_105/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_105/zeros_1Fill lstm_105/zeros_1/packed:output:0lstm_105/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_105/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_105/transpose	Transposeinputs lstm_105/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_105/Shape_1Shapelstm_105/transpose:y:0*
T0*
_output_shapes
:h
lstm_105/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_105/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_105/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_slice_1StridedSlicelstm_105/Shape_1:output:0'lstm_105/strided_slice_1/stack:output:0)lstm_105/strided_slice_1/stack_1:output:0)lstm_105/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_105/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_105/TensorArrayV2TensorListReserve-lstm_105/TensorArrayV2/element_shape:output:0!lstm_105/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_105/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_105/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_105/transpose:y:0Glstm_105/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_105/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_105/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_105/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_slice_2StridedSlicelstm_105/transpose:y:0'lstm_105/strided_slice_2/stack:output:0)lstm_105/strided_slice_2/stack_1:output:0)lstm_105/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_105/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp5lstm_105_lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_105/lstm_cell_110/MatMulMatMul!lstm_105/strided_slice_2:output:04lstm_105/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp7lstm_105_lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_105/lstm_cell_110/MatMul_1MatMullstm_105/zeros:output:06lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_105/lstm_cell_110/addAddV2'lstm_105/lstm_cell_110/MatMul:product:0)lstm_105/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp6lstm_105_lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_105/lstm_cell_110/BiasAddBiasAddlstm_105/lstm_cell_110/add:z:05lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_105/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/lstm_cell_110/splitSplit/lstm_105/lstm_cell_110/split/split_dim:output:0'lstm_105/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_105/lstm_cell_110/SigmoidSigmoid%lstm_105/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:����������
 lstm_105/lstm_cell_110/Sigmoid_1Sigmoid%lstm_105/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/mulMul$lstm_105/lstm_cell_110/Sigmoid_1:y:0lstm_105/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_105/lstm_cell_110/ReluRelu%lstm_105/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/mul_1Mul"lstm_105/lstm_cell_110/Sigmoid:y:0)lstm_105/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/add_1AddV2lstm_105/lstm_cell_110/mul:z:0 lstm_105/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_105/lstm_cell_110/Sigmoid_2Sigmoid%lstm_105/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������y
lstm_105/lstm_cell_110/Relu_1Relu lstm_105/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_105/lstm_cell_110/mul_2Mul$lstm_105/lstm_cell_110/Sigmoid_2:y:0+lstm_105/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_105/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_105/TensorArrayV2_1TensorListReserve/lstm_105/TensorArrayV2_1/element_shape:output:0!lstm_105/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_105/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_105/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_105/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_105/whileWhile$lstm_105/while/loop_counter:output:0*lstm_105/while/maximum_iterations:output:0lstm_105/time:output:0!lstm_105/TensorArrayV2_1:handle:0lstm_105/zeros:output:0lstm_105/zeros_1:output:0!lstm_105/strided_slice_1:output:0@lstm_105/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_105_lstm_cell_110_matmul_readvariableop_resource7lstm_105_lstm_cell_110_matmul_1_readvariableop_resource6lstm_105_lstm_cell_110_biasadd_readvariableop_resource*
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
lstm_105_while_body_23133631*(
cond R
lstm_105_while_cond_23133630*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_105/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_105/TensorArrayV2Stack/TensorListStackTensorListStacklstm_105/while:output:3Blstm_105/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_105/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_105/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_105/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_105/strided_slice_3StridedSlice4lstm_105/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_105/strided_slice_3/stack:output:0)lstm_105/strided_slice_3/stack_1:output:0)lstm_105/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_105/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_105/transpose_1	Transpose4lstm_105/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_105/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_105/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_106/ShapeShapelstm_105/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_106/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_106/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_106/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_sliceStridedSlicelstm_106/Shape:output:0%lstm_106/strided_slice/stack:output:0'lstm_106/strided_slice/stack_1:output:0'lstm_106/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_106/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/zeros/packedPacklstm_106/strided_slice:output:0 lstm_106/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_106/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_106/zerosFilllstm_106/zeros/packed:output:0lstm_106/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_106/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/zeros_1/packedPacklstm_106/strided_slice:output:0"lstm_106/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_106/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_106/zeros_1Fill lstm_106/zeros_1/packed:output:0lstm_106/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_106/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_106/transpose	Transposelstm_105/transpose_1:y:0 lstm_106/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_106/Shape_1Shapelstm_106/transpose:y:0*
T0*
_output_shapes
:h
lstm_106/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_106/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_106/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_slice_1StridedSlicelstm_106/Shape_1:output:0'lstm_106/strided_slice_1/stack:output:0)lstm_106/strided_slice_1/stack_1:output:0)lstm_106/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_106/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_106/TensorArrayV2TensorListReserve-lstm_106/TensorArrayV2/element_shape:output:0!lstm_106/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_106/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_106/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_106/transpose:y:0Glstm_106/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_106/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_106/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_106/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_slice_2StridedSlicelstm_106/transpose:y:0'lstm_106/strided_slice_2/stack:output:0)lstm_106/strided_slice_2/stack_1:output:0)lstm_106/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_106/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp5lstm_106_lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_106/lstm_cell_111/MatMulMatMul!lstm_106/strided_slice_2:output:04lstm_106/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp7lstm_106_lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_106/lstm_cell_111/MatMul_1MatMullstm_106/zeros:output:06lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_106/lstm_cell_111/addAddV2'lstm_106/lstm_cell_111/MatMul:product:0)lstm_106/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp6lstm_106_lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_106/lstm_cell_111/BiasAddBiasAddlstm_106/lstm_cell_111/add:z:05lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_106/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/lstm_cell_111/splitSplit/lstm_106/lstm_cell_111/split/split_dim:output:0'lstm_106/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_106/lstm_cell_111/SigmoidSigmoid%lstm_106/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:����������
 lstm_106/lstm_cell_111/Sigmoid_1Sigmoid%lstm_106/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/mulMul$lstm_106/lstm_cell_111/Sigmoid_1:y:0lstm_106/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_106/lstm_cell_111/ReluRelu%lstm_106/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/mul_1Mul"lstm_106/lstm_cell_111/Sigmoid:y:0)lstm_106/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/add_1AddV2lstm_106/lstm_cell_111/mul:z:0 lstm_106/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_106/lstm_cell_111/Sigmoid_2Sigmoid%lstm_106/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������y
lstm_106/lstm_cell_111/Relu_1Relu lstm_106/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_106/lstm_cell_111/mul_2Mul$lstm_106/lstm_cell_111/Sigmoid_2:y:0+lstm_106/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_106/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
lstm_106/TensorArrayV2_1TensorListReserve/lstm_106/TensorArrayV2_1/element_shape:output:0!lstm_106/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_106/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_106/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_106/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_106/whileWhile$lstm_106/while/loop_counter:output:0*lstm_106/while/maximum_iterations:output:0lstm_106/time:output:0!lstm_106/TensorArrayV2_1:handle:0lstm_106/zeros:output:0lstm_106/zeros_1:output:0!lstm_106/strided_slice_1:output:0@lstm_106/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_106_lstm_cell_111_matmul_readvariableop_resource7lstm_106_lstm_cell_111_matmul_1_readvariableop_resource6lstm_106_lstm_cell_111_biasadd_readvariableop_resource*
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
lstm_106_while_body_23133770*(
cond R
lstm_106_while_cond_23133769*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_106/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_106/TensorArrayV2Stack/TensorListStackTensorListStacklstm_106/while:output:3Blstm_106/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0q
lstm_106/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_106/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_106/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_106/strided_slice_3StridedSlice4lstm_106/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_106/strided_slice_3/stack:output:0)lstm_106/strided_slice_3/stack_1:output:0)lstm_106/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_106/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_106/transpose_1	Transpose4lstm_106/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_106/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_106/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_107/ShapeShapelstm_106/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_107/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_107/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_107/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_sliceStridedSlicelstm_107/Shape:output:0%lstm_107/strided_slice/stack:output:0'lstm_107/strided_slice/stack_1:output:0'lstm_107/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
lstm_107/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/zeros/packedPacklstm_107/strided_slice:output:0 lstm_107/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_107/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_107/zerosFilllstm_107/zeros/packed:output:0lstm_107/zeros/Const:output:0*
T0*'
_output_shapes
:���������[
lstm_107/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/zeros_1/packedPacklstm_107/strided_slice:output:0"lstm_107/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_107/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
lstm_107/zeros_1Fill lstm_107/zeros_1/packed:output:0lstm_107/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������l
lstm_107/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_107/transpose	Transposelstm_106/transpose_1:y:0 lstm_107/transpose/perm:output:0*
T0*+
_output_shapes
:���������V
lstm_107/Shape_1Shapelstm_107/transpose:y:0*
T0*
_output_shapes
:h
lstm_107/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_107/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_107/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_slice_1StridedSlicelstm_107/Shape_1:output:0'lstm_107/strided_slice_1/stack:output:0)lstm_107/strided_slice_1/stack_1:output:0)lstm_107/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_107/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
lstm_107/TensorArrayV2TensorListReserve-lstm_107/TensorArrayV2/element_shape:output:0!lstm_107/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
>lstm_107/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
0lstm_107/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_107/transpose:y:0Glstm_107/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���h
lstm_107/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_107/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_107/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_slice_2StridedSlicelstm_107/transpose:y:0'lstm_107/strided_slice_2/stack:output:0)lstm_107/strided_slice_2/stack_1:output:0)lstm_107/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
,lstm_107/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp5lstm_107_lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_107/lstm_cell_112/MatMulMatMul!lstm_107/strided_slice_2:output:04lstm_107/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp7lstm_107_lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_107/lstm_cell_112/MatMul_1MatMullstm_107/zeros:output:06lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_107/lstm_cell_112/addAddV2'lstm_107/lstm_cell_112/MatMul:product:0)lstm_107/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp6lstm_107_lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_107/lstm_cell_112/BiasAddBiasAddlstm_107/lstm_cell_112/add:z:05lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xh
&lstm_107/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/lstm_cell_112/splitSplit/lstm_107/lstm_cell_112/split/split_dim:output:0'lstm_107/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
lstm_107/lstm_cell_112/SigmoidSigmoid%lstm_107/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:����������
 lstm_107/lstm_cell_112/Sigmoid_1Sigmoid%lstm_107/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/mulMul$lstm_107/lstm_cell_112/Sigmoid_1:y:0lstm_107/zeros_1:output:0*
T0*'
_output_shapes
:���������|
lstm_107/lstm_cell_112/ReluRelu%lstm_107/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/mul_1Mul"lstm_107/lstm_cell_112/Sigmoid:y:0)lstm_107/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/add_1AddV2lstm_107/lstm_cell_112/mul:z:0 lstm_107/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:����������
 lstm_107/lstm_cell_112/Sigmoid_2Sigmoid%lstm_107/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������y
lstm_107/lstm_cell_112/Relu_1Relu lstm_107/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_107/lstm_cell_112/mul_2Mul$lstm_107/lstm_cell_112/Sigmoid_2:y:0+lstm_107/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������w
&lstm_107/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
%lstm_107/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_107/TensorArrayV2_1TensorListReserve/lstm_107/TensorArrayV2_1/element_shape:output:0.lstm_107/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���O
lstm_107/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_107/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
lstm_107/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
lstm_107/whileWhile$lstm_107/while/loop_counter:output:0*lstm_107/while/maximum_iterations:output:0lstm_107/time:output:0!lstm_107/TensorArrayV2_1:handle:0lstm_107/zeros:output:0lstm_107/zeros_1:output:0!lstm_107/strided_slice_1:output:0@lstm_107/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_107_lstm_cell_112_matmul_readvariableop_resource7lstm_107_lstm_cell_112_matmul_1_readvariableop_resource6lstm_107_lstm_cell_112_biasadd_readvariableop_resource*
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
lstm_107_while_body_23133910*(
cond R
lstm_107_while_cond_23133909*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
9lstm_107/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+lstm_107/TensorArrayV2Stack/TensorListStackTensorListStacklstm_107/while:output:3Blstm_107/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elementsq
lstm_107/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
 lstm_107/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_107/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
lstm_107/strided_slice_3StridedSlice4lstm_107/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_107/strided_slice_3/stack:output:0)lstm_107/strided_slice_3/stack_1:output:0)lstm_107/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskn
lstm_107/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
lstm_107/transpose_1	Transpose4lstm_107/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_107/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������d
lstm_107/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_68/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_68/dropout/MulMul!lstm_107/strided_slice_3:output:0!dropout_68/dropout/Const:output:0*
T0*'
_output_shapes
:���������i
dropout_68/dropout/ShapeShape!lstm_107/strided_slice_3:output:0*
T0*
_output_shapes
:�
/dropout_68/dropout/random_uniform/RandomUniformRandomUniform!dropout_68/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0f
!dropout_68/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_68/dropout/GreaterEqualGreaterEqual8dropout_68/dropout/random_uniform/RandomUniform:output:0*dropout_68/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������_
dropout_68/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_68/dropout/SelectV2SelectV2#dropout_68/dropout/GreaterEqual:z:0dropout_68/dropout/Mul:z:0#dropout_68/dropout/Const_1:output:0*
T0*'
_output_shapes
:����������
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_85/MatMulMatMul$dropout_68/dropout/SelectV2:output:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_85/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp.^lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp-^lstm_105/lstm_cell_110/MatMul/ReadVariableOp/^lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp^lstm_105/while.^lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp-^lstm_106/lstm_cell_111/MatMul/ReadVariableOp/^lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp^lstm_106/while.^lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp-^lstm_107/lstm_cell_112/MatMul/ReadVariableOp/^lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp^lstm_107/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2^
-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp-lstm_105/lstm_cell_110/BiasAdd/ReadVariableOp2\
,lstm_105/lstm_cell_110/MatMul/ReadVariableOp,lstm_105/lstm_cell_110/MatMul/ReadVariableOp2`
.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp.lstm_105/lstm_cell_110/MatMul_1/ReadVariableOp2 
lstm_105/whilelstm_105/while2^
-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp-lstm_106/lstm_cell_111/BiasAdd/ReadVariableOp2\
,lstm_106/lstm_cell_111/MatMul/ReadVariableOp,lstm_106/lstm_cell_111/MatMul/ReadVariableOp2`
.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp.lstm_106/lstm_cell_111/MatMul_1/ReadVariableOp2 
lstm_106/whilelstm_106/while2^
-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp-lstm_107/lstm_cell_112/BiasAdd/ReadVariableOp2\
,lstm_107/lstm_cell_112/MatMul/ReadVariableOp,lstm_107/lstm_cell_112/MatMul/ReadVariableOp2`
.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp.lstm_107/lstm_cell_112/MatMul_1/ReadVariableOp2 
lstm_107/whilelstm_107/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
0__inference_sequential_87_layer_call_fn_23132995
lstm_105_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_105_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23132943o
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
_user_specified_namelstm_105_input
�#
�
while_body_23131222
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_111_23131246_0:x0
while_lstm_cell_111_23131248_0:x,
while_lstm_cell_111_23131250_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_111_23131246:x.
while_lstm_cell_111_23131248:x*
while_lstm_cell_111_23131250:x��+while/lstm_cell_111/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
+while/lstm_cell_111/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_111_23131246_0while_lstm_cell_111_23131248_0while_lstm_cell_111_23131250_0*
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131208�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_111/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity4while/lstm_cell_111/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity4while/lstm_cell_111/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������z

while/NoOpNoOp,^while/lstm_cell_111/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_111_23131246while_lstm_cell_111_23131246_0">
while_lstm_cell_111_23131248while_lstm_cell_111_23131248_0">
while_lstm_cell_111_23131250while_lstm_cell_111_23131250_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2Z
+while/lstm_cell_111/StatefulPartitionedCall+while/lstm_cell_111/StatefulPartitionedCall: 
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
0__inference_lstm_cell_111_layer_call_fn_23136043

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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131354o
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
��
�
$__inference__traced_restore_23136472
file_prefix2
 assignvariableop_dense_85_kernel:.
 assignvariableop_1_dense_85_bias:B
0assignvariableop_2_lstm_105_lstm_cell_110_kernel:xL
:assignvariableop_3_lstm_105_lstm_cell_110_recurrent_kernel:x<
.assignvariableop_4_lstm_105_lstm_cell_110_bias:xB
0assignvariableop_5_lstm_106_lstm_cell_111_kernel:xL
:assignvariableop_6_lstm_106_lstm_cell_111_recurrent_kernel:x<
.assignvariableop_7_lstm_106_lstm_cell_111_bias:xB
0assignvariableop_8_lstm_107_lstm_cell_112_kernel:xL
:assignvariableop_9_lstm_107_lstm_cell_112_recurrent_kernel:x=
/assignvariableop_10_lstm_107_lstm_cell_112_bias:x'
assignvariableop_11_iteration:	 +
!assignvariableop_12_learning_rate: J
8assignvariableop_13_adam_m_lstm_105_lstm_cell_110_kernel:xJ
8assignvariableop_14_adam_v_lstm_105_lstm_cell_110_kernel:xT
Bassignvariableop_15_adam_m_lstm_105_lstm_cell_110_recurrent_kernel:xT
Bassignvariableop_16_adam_v_lstm_105_lstm_cell_110_recurrent_kernel:xD
6assignvariableop_17_adam_m_lstm_105_lstm_cell_110_bias:xD
6assignvariableop_18_adam_v_lstm_105_lstm_cell_110_bias:xJ
8assignvariableop_19_adam_m_lstm_106_lstm_cell_111_kernel:xJ
8assignvariableop_20_adam_v_lstm_106_lstm_cell_111_kernel:xT
Bassignvariableop_21_adam_m_lstm_106_lstm_cell_111_recurrent_kernel:xT
Bassignvariableop_22_adam_v_lstm_106_lstm_cell_111_recurrent_kernel:xD
6assignvariableop_23_adam_m_lstm_106_lstm_cell_111_bias:xD
6assignvariableop_24_adam_v_lstm_106_lstm_cell_111_bias:xJ
8assignvariableop_25_adam_m_lstm_107_lstm_cell_112_kernel:xJ
8assignvariableop_26_adam_v_lstm_107_lstm_cell_112_kernel:xT
Bassignvariableop_27_adam_m_lstm_107_lstm_cell_112_recurrent_kernel:xT
Bassignvariableop_28_adam_v_lstm_107_lstm_cell_112_recurrent_kernel:xD
6assignvariableop_29_adam_m_lstm_107_lstm_cell_112_bias:xD
6assignvariableop_30_adam_v_lstm_107_lstm_cell_112_bias:x<
*assignvariableop_31_adam_m_dense_85_kernel:<
*assignvariableop_32_adam_v_dense_85_kernel:6
(assignvariableop_33_adam_m_dense_85_bias:6
(assignvariableop_34_adam_v_dense_85_bias:%
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
AssignVariableOpAssignVariableOp assignvariableop_dense_85_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_85_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_lstm_105_lstm_cell_110_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_lstm_105_lstm_cell_110_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_lstm_105_lstm_cell_110_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_lstm_106_lstm_cell_111_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp:assignvariableop_6_lstm_106_lstm_cell_111_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_106_lstm_cell_111_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_lstm_107_lstm_cell_112_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp:assignvariableop_9_lstm_107_lstm_cell_112_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_107_lstm_cell_112_biasIdentity_10:output:0"/device:CPU:0*&
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
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_m_lstm_105_lstm_cell_110_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp8assignvariableop_14_adam_v_lstm_105_lstm_cell_110_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpBassignvariableop_15_adam_m_lstm_105_lstm_cell_110_recurrent_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpBassignvariableop_16_adam_v_lstm_105_lstm_cell_110_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_m_lstm_105_lstm_cell_110_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_v_lstm_105_lstm_cell_110_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_m_lstm_106_lstm_cell_111_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_v_lstm_106_lstm_cell_111_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpBassignvariableop_21_adam_m_lstm_106_lstm_cell_111_recurrent_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adam_v_lstm_106_lstm_cell_111_recurrent_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_m_lstm_106_lstm_cell_111_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_v_lstm_106_lstm_cell_111_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_m_lstm_107_lstm_cell_112_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp8assignvariableop_26_adam_v_lstm_107_lstm_cell_112_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpBassignvariableop_27_adam_m_lstm_107_lstm_cell_112_recurrent_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_v_lstm_107_lstm_cell_112_recurrent_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_m_lstm_107_lstm_cell_112_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_v_lstm_107_lstm_cell_112_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_m_dense_85_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_v_dense_85_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_m_dense_85_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_v_dense_85_biasIdentity_34:output:0"/device:CPU:0*&
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
�
�
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136173

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
while_cond_23131221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23131221___redundant_placeholder06
2while_while_cond_23131221___redundant_placeholder16
2while_while_cond_23131221___redundant_placeholder26
2while_while_cond_23131221___redundant_placeholder3
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
�C
�

lstm_105_while_body_23133631.
*lstm_105_while_lstm_105_while_loop_counter4
0lstm_105_while_lstm_105_while_maximum_iterations
lstm_105_while_placeholder 
lstm_105_while_placeholder_1 
lstm_105_while_placeholder_2 
lstm_105_while_placeholder_3-
)lstm_105_while_lstm_105_strided_slice_1_0i
elstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0:xQ
?lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0:xL
>lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0:x
lstm_105_while_identity
lstm_105_while_identity_1
lstm_105_while_identity_2
lstm_105_while_identity_3
lstm_105_while_identity_4
lstm_105_while_identity_5+
'lstm_105_while_lstm_105_strided_slice_1g
clstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensorM
;lstm_105_while_lstm_cell_110_matmul_readvariableop_resource:xO
=lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource:xJ
<lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource:x��3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp�2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp�4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp�
@lstm_105/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_105/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensor_0lstm_105_while_placeholderIlstm_105/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp=lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_105/while/lstm_cell_110/MatMulMatMul9lstm_105/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp?lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_105/while/lstm_cell_110/MatMul_1MatMullstm_105_while_placeholder_2<lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_105/while/lstm_cell_110/addAddV2-lstm_105/while/lstm_cell_110/MatMul:product:0/lstm_105/while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp>lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_105/while/lstm_cell_110/BiasAddBiasAdd$lstm_105/while/lstm_cell_110/add:z:0;lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_105/while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_105/while/lstm_cell_110/splitSplit5lstm_105/while/lstm_cell_110/split/split_dim:output:0-lstm_105/while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_105/while/lstm_cell_110/SigmoidSigmoid+lstm_105/while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:����������
&lstm_105/while/lstm_cell_110/Sigmoid_1Sigmoid+lstm_105/while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
 lstm_105/while/lstm_cell_110/mulMul*lstm_105/while/lstm_cell_110/Sigmoid_1:y:0lstm_105_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_105/while/lstm_cell_110/ReluRelu+lstm_105/while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
"lstm_105/while/lstm_cell_110/mul_1Mul(lstm_105/while/lstm_cell_110/Sigmoid:y:0/lstm_105/while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_105/while/lstm_cell_110/add_1AddV2$lstm_105/while/lstm_cell_110/mul:z:0&lstm_105/while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_105/while/lstm_cell_110/Sigmoid_2Sigmoid+lstm_105/while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:����������
#lstm_105/while/lstm_cell_110/Relu_1Relu&lstm_105/while/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_105/while/lstm_cell_110/mul_2Mul*lstm_105/while/lstm_cell_110/Sigmoid_2:y:01lstm_105/while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_105/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_105_while_placeholder_1lstm_105_while_placeholder&lstm_105/while/lstm_cell_110/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_105/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_105/while/addAddV2lstm_105_while_placeholderlstm_105/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_105/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_105/while/add_1AddV2*lstm_105_while_lstm_105_while_loop_counterlstm_105/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_105/while/IdentityIdentitylstm_105/while/add_1:z:0^lstm_105/while/NoOp*
T0*
_output_shapes
: �
lstm_105/while/Identity_1Identity0lstm_105_while_lstm_105_while_maximum_iterations^lstm_105/while/NoOp*
T0*
_output_shapes
: t
lstm_105/while/Identity_2Identitylstm_105/while/add:z:0^lstm_105/while/NoOp*
T0*
_output_shapes
: �
lstm_105/while/Identity_3IdentityClstm_105/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_105/while/NoOp*
T0*
_output_shapes
: �
lstm_105/while/Identity_4Identity&lstm_105/while/lstm_cell_110/mul_2:z:0^lstm_105/while/NoOp*
T0*'
_output_shapes
:����������
lstm_105/while/Identity_5Identity&lstm_105/while/lstm_cell_110/add_1:z:0^lstm_105/while/NoOp*
T0*'
_output_shapes
:����������
lstm_105/while/NoOpNoOp4^lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp3^lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp5^lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_105_while_identity lstm_105/while/Identity:output:0"?
lstm_105_while_identity_1"lstm_105/while/Identity_1:output:0"?
lstm_105_while_identity_2"lstm_105/while/Identity_2:output:0"?
lstm_105_while_identity_3"lstm_105/while/Identity_3:output:0"?
lstm_105_while_identity_4"lstm_105/while/Identity_4:output:0"?
lstm_105_while_identity_5"lstm_105/while/Identity_5:output:0"T
'lstm_105_while_lstm_105_strided_slice_1)lstm_105_while_lstm_105_strided_slice_1_0"~
<lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource>lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0"�
=lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource?lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0"|
;lstm_105_while_lstm_cell_110_matmul_readvariableop_resource=lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0"�
clstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensorelstm_105_while_tensorarrayv2read_tensorlistgetitem_lstm_105_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp3lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp2h
2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp2lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp2l
4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp4lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
while_cond_23131062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23131062___redundant_placeholder06
2while_while_cond_23131062___redundant_placeholder16
2while_while_cond_23131062___redundant_placeholder26
2while_while_cond_23131062___redundant_placeholder3
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
�S
�
*sequential_87_lstm_106_while_body_23130559J
Fsequential_87_lstm_106_while_sequential_87_lstm_106_while_loop_counterP
Lsequential_87_lstm_106_while_sequential_87_lstm_106_while_maximum_iterations,
(sequential_87_lstm_106_while_placeholder.
*sequential_87_lstm_106_while_placeholder_1.
*sequential_87_lstm_106_while_placeholder_2.
*sequential_87_lstm_106_while_placeholder_3I
Esequential_87_lstm_106_while_sequential_87_lstm_106_strided_slice_1_0�
�sequential_87_lstm_106_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_106_tensorarrayunstack_tensorlistfromtensor_0]
Ksequential_87_lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0:x_
Msequential_87_lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0:xZ
Lsequential_87_lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0:x)
%sequential_87_lstm_106_while_identity+
'sequential_87_lstm_106_while_identity_1+
'sequential_87_lstm_106_while_identity_2+
'sequential_87_lstm_106_while_identity_3+
'sequential_87_lstm_106_while_identity_4+
'sequential_87_lstm_106_while_identity_5G
Csequential_87_lstm_106_while_sequential_87_lstm_106_strided_slice_1�
sequential_87_lstm_106_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_106_tensorarrayunstack_tensorlistfromtensor[
Isequential_87_lstm_106_while_lstm_cell_111_matmul_readvariableop_resource:x]
Ksequential_87_lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource:xX
Jsequential_87_lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource:x��Asequential_87/lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp�@sequential_87/lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp�Bsequential_87/lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp�
Nsequential_87/lstm_106/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
@sequential_87/lstm_106/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_87_lstm_106_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_106_tensorarrayunstack_tensorlistfromtensor_0(sequential_87_lstm_106_while_placeholderWsequential_87/lstm_106/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
@sequential_87/lstm_106/while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOpKsequential_87_lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
1sequential_87/lstm_106/while/lstm_cell_111/MatMulMatMulGsequential_87/lstm_106/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_87/lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
Bsequential_87/lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOpMsequential_87_lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
3sequential_87/lstm_106/while/lstm_cell_111/MatMul_1MatMul*sequential_87_lstm_106_while_placeholder_2Jsequential_87/lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.sequential_87/lstm_106/while/lstm_cell_111/addAddV2;sequential_87/lstm_106/while/lstm_cell_111/MatMul:product:0=sequential_87/lstm_106/while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
Asequential_87/lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOpLsequential_87_lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
2sequential_87/lstm_106/while/lstm_cell_111/BiasAddBiasAdd2sequential_87/lstm_106/while/lstm_cell_111/add:z:0Isequential_87/lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x|
:sequential_87/lstm_106/while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_87/lstm_106/while/lstm_cell_111/splitSplitCsequential_87/lstm_106/while/lstm_cell_111/split/split_dim:output:0;sequential_87/lstm_106/while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
2sequential_87/lstm_106/while/lstm_cell_111/SigmoidSigmoid9sequential_87/lstm_106/while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_106/while/lstm_cell_111/Sigmoid_1Sigmoid9sequential_87/lstm_106/while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
.sequential_87/lstm_106/while/lstm_cell_111/mulMul8sequential_87/lstm_106/while/lstm_cell_111/Sigmoid_1:y:0*sequential_87_lstm_106_while_placeholder_3*
T0*'
_output_shapes
:����������
/sequential_87/lstm_106/while/lstm_cell_111/ReluRelu9sequential_87/lstm_106/while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
0sequential_87/lstm_106/while/lstm_cell_111/mul_1Mul6sequential_87/lstm_106/while/lstm_cell_111/Sigmoid:y:0=sequential_87/lstm_106/while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
0sequential_87/lstm_106/while/lstm_cell_111/add_1AddV22sequential_87/lstm_106/while/lstm_cell_111/mul:z:04sequential_87/lstm_106/while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_106/while/lstm_cell_111/Sigmoid_2Sigmoid9sequential_87/lstm_106/while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:����������
1sequential_87/lstm_106/while/lstm_cell_111/Relu_1Relu4sequential_87/lstm_106/while/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
0sequential_87/lstm_106/while/lstm_cell_111/mul_2Mul8sequential_87/lstm_106/while/lstm_cell_111/Sigmoid_2:y:0?sequential_87/lstm_106/while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
Asequential_87/lstm_106/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_87_lstm_106_while_placeholder_1(sequential_87_lstm_106_while_placeholder4sequential_87/lstm_106/while/lstm_cell_111/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"sequential_87/lstm_106/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_87/lstm_106/while/addAddV2(sequential_87_lstm_106_while_placeholder+sequential_87/lstm_106/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_87/lstm_106/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"sequential_87/lstm_106/while/add_1AddV2Fsequential_87_lstm_106_while_sequential_87_lstm_106_while_loop_counter-sequential_87/lstm_106/while/add_1/y:output:0*
T0*
_output_shapes
: �
%sequential_87/lstm_106/while/IdentityIdentity&sequential_87/lstm_106/while/add_1:z:0"^sequential_87/lstm_106/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_106/while/Identity_1IdentityLsequential_87_lstm_106_while_sequential_87_lstm_106_while_maximum_iterations"^sequential_87/lstm_106/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_106/while/Identity_2Identity$sequential_87/lstm_106/while/add:z:0"^sequential_87/lstm_106/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_106/while/Identity_3IdentityQsequential_87/lstm_106/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_87/lstm_106/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_106/while/Identity_4Identity4sequential_87/lstm_106/while/lstm_cell_111/mul_2:z:0"^sequential_87/lstm_106/while/NoOp*
T0*'
_output_shapes
:����������
'sequential_87/lstm_106/while/Identity_5Identity4sequential_87/lstm_106/while/lstm_cell_111/add_1:z:0"^sequential_87/lstm_106/while/NoOp*
T0*'
_output_shapes
:����������
!sequential_87/lstm_106/while/NoOpNoOpB^sequential_87/lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOpA^sequential_87/lstm_106/while/lstm_cell_111/MatMul/ReadVariableOpC^sequential_87/lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_87_lstm_106_while_identity.sequential_87/lstm_106/while/Identity:output:0"[
'sequential_87_lstm_106_while_identity_10sequential_87/lstm_106/while/Identity_1:output:0"[
'sequential_87_lstm_106_while_identity_20sequential_87/lstm_106/while/Identity_2:output:0"[
'sequential_87_lstm_106_while_identity_30sequential_87/lstm_106/while/Identity_3:output:0"[
'sequential_87_lstm_106_while_identity_40sequential_87/lstm_106/while/Identity_4:output:0"[
'sequential_87_lstm_106_while_identity_50sequential_87/lstm_106/while/Identity_5:output:0"�
Jsequential_87_lstm_106_while_lstm_cell_111_biasadd_readvariableop_resourceLsequential_87_lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0"�
Ksequential_87_lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resourceMsequential_87_lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0"�
Isequential_87_lstm_106_while_lstm_cell_111_matmul_readvariableop_resourceKsequential_87_lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0"�
Csequential_87_lstm_106_while_sequential_87_lstm_106_strided_slice_1Esequential_87_lstm_106_while_sequential_87_lstm_106_strided_slice_1_0"�
sequential_87_lstm_106_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_106_tensorarrayunstack_tensorlistfromtensor�sequential_87_lstm_106_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_106_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2�
Asequential_87/lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOpAsequential_87/lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp2�
@sequential_87/lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp@sequential_87/lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp2�
Bsequential_87/lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOpBsequential_87/lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
while_cond_23132210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23132210___redundant_placeholder06
2while_while_cond_23132210___redundant_placeholder16
2while_while_cond_23132210___redundant_placeholder26
2while_while_cond_23132210___redundant_placeholder3
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
�L
�
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135575
inputs_0>
,lstm_cell_112_matmul_readvariableop_resource:x@
.lstm_cell_112_matmul_1_readvariableop_resource:x;
-lstm_cell_112_biasadd_readvariableop_resource:x
identity��$lstm_cell_112/BiasAdd/ReadVariableOp�#lstm_cell_112/MatMul/ReadVariableOp�%lstm_cell_112/MatMul_1/ReadVariableOp�while=
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
#lstm_cell_112/MatMul/ReadVariableOpReadVariableOp,lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMulMatMulstrided_slice_2:output:0+lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMul_1MatMulzeros:output:0-lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_112/addAddV2lstm_cell_112/MatMul:product:0 lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_112/BiasAddBiasAddlstm_cell_112/add:z:0,lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_112/splitSplit&lstm_cell_112/split/split_dim:output:0lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_112/SigmoidSigmoidlstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_1Sigmoidlstm_cell_112/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_112/mulMullstm_cell_112/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_112/ReluRelulstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_1Mullstm_cell_112/Sigmoid:y:0 lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_112/add_1AddV2lstm_cell_112/mul:z:0lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_2Sigmoidlstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_112/Relu_1Relulstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_2Mullstm_cell_112/Sigmoid_2:y:0"lstm_cell_112/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_112_matmul_readvariableop_resource.lstm_cell_112_matmul_1_readvariableop_resource-lstm_cell_112_biasadd_readvariableop_resource*
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
while_body_23135490*
condR
while_cond_23135489*K
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
NoOpNoOp%^lstm_cell_112/BiasAdd/ReadVariableOp$^lstm_cell_112/MatMul/ReadVariableOp&^lstm_cell_112/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_112/BiasAdd/ReadVariableOp$lstm_cell_112/BiasAdd/ReadVariableOp2J
#lstm_cell_112/MatMul/ReadVariableOp#lstm_cell_112/MatMul/ReadVariableOp2N
%lstm_cell_112/MatMul_1/ReadVariableOp%lstm_cell_112/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
+__inference_lstm_105_layer_call_fn_23134031
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23131132|
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
while_body_23134871
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_111_matmul_readvariableop_resource_0:xH
6while_lstm_cell_111_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_111_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_111_matmul_readvariableop_resource:xF
4while_lstm_cell_111_matmul_1_readvariableop_resource:xA
3while_lstm_cell_111_biasadd_readvariableop_resource:x��*while/lstm_cell_111/BiasAdd/ReadVariableOp�)while/lstm_cell_111/MatMul/ReadVariableOp�+while/lstm_cell_111/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_111/addAddV2$while/lstm_cell_111/MatMul:product:0&while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_111/BiasAddBiasAddwhile/lstm_cell_111/add:z:02while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_111/splitSplit,while/lstm_cell_111/split/split_dim:output:0$while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_111/SigmoidSigmoid"while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_1Sigmoid"while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mulMul!while/lstm_cell_111/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_111/ReluRelu"while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_1Mulwhile/lstm_cell_111/Sigmoid:y:0&while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/add_1AddV2while/lstm_cell_111/mul:z:0while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_2Sigmoid"while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_111/Relu_1Reluwhile/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_2Mul!while/lstm_cell_111/Sigmoid_2:y:0(while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_111/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_111/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_111/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_111/BiasAdd/ReadVariableOp*^while/lstm_cell_111/MatMul/ReadVariableOp,^while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_111_biasadd_readvariableop_resource5while_lstm_cell_111_biasadd_readvariableop_resource_0"n
4while_lstm_cell_111_matmul_1_readvariableop_resource6while_lstm_cell_111_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_111_matmul_readvariableop_resource4while_lstm_cell_111_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_111/BiasAdd/ReadVariableOp*while/lstm_cell_111/BiasAdd/ReadVariableOp2V
)while/lstm_cell_111/MatMul/ReadVariableOp)while/lstm_cell_111/MatMul/ReadVariableOp2Z
+while/lstm_cell_111/MatMul_1/ReadVariableOp+while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
while_cond_23132789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23132789___redundant_placeholder06
2while_while_cond_23132789___redundant_placeholder16
2while_while_cond_23132789___redundant_placeholder26
2while_while_cond_23132789___redundant_placeholder3
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131208

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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136107

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
�L
�
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135430
inputs_0>
,lstm_cell_112_matmul_readvariableop_resource:x@
.lstm_cell_112_matmul_1_readvariableop_resource:x;
-lstm_cell_112_biasadd_readvariableop_resource:x
identity��$lstm_cell_112/BiasAdd/ReadVariableOp�#lstm_cell_112/MatMul/ReadVariableOp�%lstm_cell_112/MatMul_1/ReadVariableOp�while=
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
#lstm_cell_112/MatMul/ReadVariableOpReadVariableOp,lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMulMatMulstrided_slice_2:output:0+lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMul_1MatMulzeros:output:0-lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_112/addAddV2lstm_cell_112/MatMul:product:0 lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_112/BiasAddBiasAddlstm_cell_112/add:z:0,lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_112/splitSplit&lstm_cell_112/split/split_dim:output:0lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_112/SigmoidSigmoidlstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_1Sigmoidlstm_cell_112/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_112/mulMullstm_cell_112/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_112/ReluRelulstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_1Mullstm_cell_112/Sigmoid:y:0 lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_112/add_1AddV2lstm_cell_112/mul:z:0lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_2Sigmoidlstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_112/Relu_1Relulstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_2Mullstm_cell_112/Sigmoid_2:y:0"lstm_cell_112/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_112_matmul_readvariableop_resource.lstm_cell_112_matmul_1_readvariableop_resource-lstm_cell_112_biasadd_readvariableop_resource*
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
while_body_23135345*
condR
while_cond_23135344*K
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
NoOpNoOp%^lstm_cell_112/BiasAdd/ReadVariableOp$^lstm_cell_112/MatMul/ReadVariableOp&^lstm_cell_112/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_112/BiasAdd/ReadVariableOp$lstm_cell_112/BiasAdd/ReadVariableOp2J
#lstm_cell_112/MatMul/ReadVariableOp#lstm_cell_112/MatMul/ReadVariableOp2N
%lstm_cell_112/MatMul_1/ReadVariableOp%lstm_cell_112/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
while_cond_23132458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23132458___redundant_placeholder06
2while_while_cond_23132458___redundant_placeholder16
2while_while_cond_23132458___redundant_placeholder26
2while_while_cond_23132458___redundant_placeholder3
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
�
K__inference_sequential_87_layer_call_and_return_conditional_losses_23132943

inputs#
lstm_105_23132915:x#
lstm_105_23132917:x
lstm_105_23132919:x#
lstm_106_23132922:x#
lstm_106_23132924:x
lstm_106_23132926:x#
lstm_107_23132929:x#
lstm_107_23132931:x
lstm_107_23132933:x#
dense_85_23132937:
dense_85_23132939:
identity�� dense_85/StatefulPartitionedCall�"dropout_68/StatefulPartitionedCall� lstm_105/StatefulPartitionedCall� lstm_106/StatefulPartitionedCall� lstm_107/StatefulPartitionedCall�
 lstm_105/StatefulPartitionedCallStatefulPartitionedCallinputslstm_105_23132915lstm_105_23132917lstm_105_23132919*
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23132874�
 lstm_106/StatefulPartitionedCallStatefulPartitionedCall)lstm_105/StatefulPartitionedCall:output:0lstm_106_23132922lstm_106_23132924lstm_106_23132926*
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132709�
 lstm_107/StatefulPartitionedCallStatefulPartitionedCall)lstm_106/StatefulPartitionedCall:output:0lstm_107_23132929lstm_107_23132931lstm_107_23132933*
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132544�
"dropout_68/StatefulPartitionedCallStatefulPartitionedCall)lstm_107/StatefulPartitionedCall:output:0*
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132383�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall+dropout_68/StatefulPartitionedCall:output:0dense_85_23132937dense_85_23132939*
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
F__inference_dense_85_layer_call_and_return_conditional_losses_23132321x
IdentityIdentity)dense_85/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_85/StatefulPartitionedCall#^dropout_68/StatefulPartitionedCall!^lstm_105/StatefulPartitionedCall!^lstm_106/StatefulPartitionedCall!^lstm_107/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2H
"dropout_68/StatefulPartitionedCall"dropout_68/StatefulPartitionedCall2D
 lstm_105/StatefulPartitionedCall lstm_105/StatefulPartitionedCall2D
 lstm_106/StatefulPartitionedCall lstm_106/StatefulPartitionedCall2D
 lstm_107/StatefulPartitionedCall lstm_107/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
&__inference_signature_wrapper_23133088
lstm_105_input
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
StatefulPartitionedCallStatefulPartitionedCalllstm_105_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
#__inference__wrapped_model_23130791o
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
_user_specified_namelstm_105_input
�S
�
*sequential_87_lstm_105_while_body_23130420J
Fsequential_87_lstm_105_while_sequential_87_lstm_105_while_loop_counterP
Lsequential_87_lstm_105_while_sequential_87_lstm_105_while_maximum_iterations,
(sequential_87_lstm_105_while_placeholder.
*sequential_87_lstm_105_while_placeholder_1.
*sequential_87_lstm_105_while_placeholder_2.
*sequential_87_lstm_105_while_placeholder_3I
Esequential_87_lstm_105_while_sequential_87_lstm_105_strided_slice_1_0�
�sequential_87_lstm_105_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_105_tensorarrayunstack_tensorlistfromtensor_0]
Ksequential_87_lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0:x_
Msequential_87_lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0:xZ
Lsequential_87_lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0:x)
%sequential_87_lstm_105_while_identity+
'sequential_87_lstm_105_while_identity_1+
'sequential_87_lstm_105_while_identity_2+
'sequential_87_lstm_105_while_identity_3+
'sequential_87_lstm_105_while_identity_4+
'sequential_87_lstm_105_while_identity_5G
Csequential_87_lstm_105_while_sequential_87_lstm_105_strided_slice_1�
sequential_87_lstm_105_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_105_tensorarrayunstack_tensorlistfromtensor[
Isequential_87_lstm_105_while_lstm_cell_110_matmul_readvariableop_resource:x]
Ksequential_87_lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource:xX
Jsequential_87_lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource:x��Asequential_87/lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp�@sequential_87/lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp�Bsequential_87/lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp�
Nsequential_87/lstm_105/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
@sequential_87/lstm_105/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_87_lstm_105_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_105_tensorarrayunstack_tensorlistfromtensor_0(sequential_87_lstm_105_while_placeholderWsequential_87/lstm_105/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
@sequential_87/lstm_105/while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOpKsequential_87_lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
1sequential_87/lstm_105/while/lstm_cell_110/MatMulMatMulGsequential_87/lstm_105/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_87/lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
Bsequential_87/lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOpMsequential_87_lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
3sequential_87/lstm_105/while/lstm_cell_110/MatMul_1MatMul*sequential_87_lstm_105_while_placeholder_2Jsequential_87/lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.sequential_87/lstm_105/while/lstm_cell_110/addAddV2;sequential_87/lstm_105/while/lstm_cell_110/MatMul:product:0=sequential_87/lstm_105/while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
Asequential_87/lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOpLsequential_87_lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
2sequential_87/lstm_105/while/lstm_cell_110/BiasAddBiasAdd2sequential_87/lstm_105/while/lstm_cell_110/add:z:0Isequential_87/lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x|
:sequential_87/lstm_105/while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_87/lstm_105/while/lstm_cell_110/splitSplitCsequential_87/lstm_105/while/lstm_cell_110/split/split_dim:output:0;sequential_87/lstm_105/while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
2sequential_87/lstm_105/while/lstm_cell_110/SigmoidSigmoid9sequential_87/lstm_105/while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_105/while/lstm_cell_110/Sigmoid_1Sigmoid9sequential_87/lstm_105/while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
.sequential_87/lstm_105/while/lstm_cell_110/mulMul8sequential_87/lstm_105/while/lstm_cell_110/Sigmoid_1:y:0*sequential_87_lstm_105_while_placeholder_3*
T0*'
_output_shapes
:����������
/sequential_87/lstm_105/while/lstm_cell_110/ReluRelu9sequential_87/lstm_105/while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
0sequential_87/lstm_105/while/lstm_cell_110/mul_1Mul6sequential_87/lstm_105/while/lstm_cell_110/Sigmoid:y:0=sequential_87/lstm_105/while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
0sequential_87/lstm_105/while/lstm_cell_110/add_1AddV22sequential_87/lstm_105/while/lstm_cell_110/mul:z:04sequential_87/lstm_105/while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_105/while/lstm_cell_110/Sigmoid_2Sigmoid9sequential_87/lstm_105/while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:����������
1sequential_87/lstm_105/while/lstm_cell_110/Relu_1Relu4sequential_87/lstm_105/while/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
0sequential_87/lstm_105/while/lstm_cell_110/mul_2Mul8sequential_87/lstm_105/while/lstm_cell_110/Sigmoid_2:y:0?sequential_87/lstm_105/while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
Asequential_87/lstm_105/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_87_lstm_105_while_placeholder_1(sequential_87_lstm_105_while_placeholder4sequential_87/lstm_105/while/lstm_cell_110/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"sequential_87/lstm_105/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_87/lstm_105/while/addAddV2(sequential_87_lstm_105_while_placeholder+sequential_87/lstm_105/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_87/lstm_105/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"sequential_87/lstm_105/while/add_1AddV2Fsequential_87_lstm_105_while_sequential_87_lstm_105_while_loop_counter-sequential_87/lstm_105/while/add_1/y:output:0*
T0*
_output_shapes
: �
%sequential_87/lstm_105/while/IdentityIdentity&sequential_87/lstm_105/while/add_1:z:0"^sequential_87/lstm_105/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_105/while/Identity_1IdentityLsequential_87_lstm_105_while_sequential_87_lstm_105_while_maximum_iterations"^sequential_87/lstm_105/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_105/while/Identity_2Identity$sequential_87/lstm_105/while/add:z:0"^sequential_87/lstm_105/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_105/while/Identity_3IdentityQsequential_87/lstm_105/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_87/lstm_105/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_105/while/Identity_4Identity4sequential_87/lstm_105/while/lstm_cell_110/mul_2:z:0"^sequential_87/lstm_105/while/NoOp*
T0*'
_output_shapes
:����������
'sequential_87/lstm_105/while/Identity_5Identity4sequential_87/lstm_105/while/lstm_cell_110/add_1:z:0"^sequential_87/lstm_105/while/NoOp*
T0*'
_output_shapes
:����������
!sequential_87/lstm_105/while/NoOpNoOpB^sequential_87/lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOpA^sequential_87/lstm_105/while/lstm_cell_110/MatMul/ReadVariableOpC^sequential_87/lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_87_lstm_105_while_identity.sequential_87/lstm_105/while/Identity:output:0"[
'sequential_87_lstm_105_while_identity_10sequential_87/lstm_105/while/Identity_1:output:0"[
'sequential_87_lstm_105_while_identity_20sequential_87/lstm_105/while/Identity_2:output:0"[
'sequential_87_lstm_105_while_identity_30sequential_87/lstm_105/while/Identity_3:output:0"[
'sequential_87_lstm_105_while_identity_40sequential_87/lstm_105/while/Identity_4:output:0"[
'sequential_87_lstm_105_while_identity_50sequential_87/lstm_105/while/Identity_5:output:0"�
Jsequential_87_lstm_105_while_lstm_cell_110_biasadd_readvariableop_resourceLsequential_87_lstm_105_while_lstm_cell_110_biasadd_readvariableop_resource_0"�
Ksequential_87_lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resourceMsequential_87_lstm_105_while_lstm_cell_110_matmul_1_readvariableop_resource_0"�
Isequential_87_lstm_105_while_lstm_cell_110_matmul_readvariableop_resourceKsequential_87_lstm_105_while_lstm_cell_110_matmul_readvariableop_resource_0"�
Csequential_87_lstm_105_while_sequential_87_lstm_105_strided_slice_1Esequential_87_lstm_105_while_sequential_87_lstm_105_strided_slice_1_0"�
sequential_87_lstm_105_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_105_tensorarrayunstack_tensorlistfromtensor�sequential_87_lstm_105_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_105_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2�
Asequential_87/lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOpAsequential_87/lstm_105/while/lstm_cell_110/BiasAdd/ReadVariableOp2�
@sequential_87/lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp@sequential_87/lstm_105/while/lstm_cell_110/MatMul/ReadVariableOp2�
Bsequential_87/lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOpBsequential_87/lstm_105/while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
+__inference_lstm_105_layer_call_fn_23134020
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23130941|
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
�
+__inference_lstm_106_layer_call_fn_23134658

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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132144s
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
�K
�
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135865

inputs>
,lstm_cell_112_matmul_readvariableop_resource:x@
.lstm_cell_112_matmul_1_readvariableop_resource:x;
-lstm_cell_112_biasadd_readvariableop_resource:x
identity��$lstm_cell_112/BiasAdd/ReadVariableOp�#lstm_cell_112/MatMul/ReadVariableOp�%lstm_cell_112/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_112/MatMul/ReadVariableOpReadVariableOp,lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMulMatMulstrided_slice_2:output:0+lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMul_1MatMulzeros:output:0-lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_112/addAddV2lstm_cell_112/MatMul:product:0 lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_112/BiasAddBiasAddlstm_cell_112/add:z:0,lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_112/splitSplit&lstm_cell_112/split/split_dim:output:0lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_112/SigmoidSigmoidlstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_1Sigmoidlstm_cell_112/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_112/mulMullstm_cell_112/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_112/ReluRelulstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_1Mullstm_cell_112/Sigmoid:y:0 lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_112/add_1AddV2lstm_cell_112/mul:z:0lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_2Sigmoidlstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_112/Relu_1Relulstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_2Mullstm_cell_112/Sigmoid_2:y:0"lstm_cell_112/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_112_matmul_readvariableop_resource.lstm_cell_112_matmul_1_readvariableop_resource-lstm_cell_112_biasadd_readvariableop_resource*
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
while_body_23135780*
condR
while_cond_23135779*K
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
NoOpNoOp%^lstm_cell_112/BiasAdd/ReadVariableOp$^lstm_cell_112/MatMul/ReadVariableOp&^lstm_cell_112/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_112/BiasAdd/ReadVariableOp$lstm_cell_112/BiasAdd/ReadVariableOp2J
#lstm_cell_112/MatMul/ReadVariableOp#lstm_cell_112/MatMul/ReadVariableOp2N
%lstm_cell_112/MatMul_1/ReadVariableOp%lstm_cell_112/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�8
�
while_body_23134728
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_111_matmul_readvariableop_resource_0:xH
6while_lstm_cell_111_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_111_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_111_matmul_readvariableop_resource:xF
4while_lstm_cell_111_matmul_1_readvariableop_resource:xA
3while_lstm_cell_111_biasadd_readvariableop_resource:x��*while/lstm_cell_111/BiasAdd/ReadVariableOp�)while/lstm_cell_111/MatMul/ReadVariableOp�+while/lstm_cell_111/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_111/addAddV2$while/lstm_cell_111/MatMul:product:0&while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_111/BiasAddBiasAddwhile/lstm_cell_111/add:z:02while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_111/splitSplit,while/lstm_cell_111/split/split_dim:output:0$while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_111/SigmoidSigmoid"while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_1Sigmoid"while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mulMul!while/lstm_cell_111/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_111/ReluRelu"while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_1Mulwhile/lstm_cell_111/Sigmoid:y:0&while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/add_1AddV2while/lstm_cell_111/mul:z:0while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_2Sigmoid"while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_111/Relu_1Reluwhile/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_2Mul!while/lstm_cell_111/Sigmoid_2:y:0(while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_111/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_111/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_111/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_111/BiasAdd/ReadVariableOp*^while/lstm_cell_111/MatMul/ReadVariableOp,^while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_111_biasadd_readvariableop_resource5while_lstm_cell_111_biasadd_readvariableop_resource_0"n
4while_lstm_cell_111_matmul_1_readvariableop_resource6while_lstm_cell_111_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_111_matmul_readvariableop_resource4while_lstm_cell_111_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_111/BiasAdd/ReadVariableOp*while/lstm_cell_111/BiasAdd/ReadVariableOp2V
)while/lstm_cell_111/MatMul/ReadVariableOp)while/lstm_cell_111/MatMul/ReadVariableOp2Z
+while/lstm_cell_111/MatMul_1/ReadVariableOp+while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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

lstm_106_while_body_23133770.
*lstm_106_while_lstm_106_while_loop_counter4
0lstm_106_while_lstm_106_while_maximum_iterations
lstm_106_while_placeholder 
lstm_106_while_placeholder_1 
lstm_106_while_placeholder_2 
lstm_106_while_placeholder_3-
)lstm_106_while_lstm_106_strided_slice_1_0i
elstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensor_0O
=lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0:xQ
?lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0:xL
>lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0:x
lstm_106_while_identity
lstm_106_while_identity_1
lstm_106_while_identity_2
lstm_106_while_identity_3
lstm_106_while_identity_4
lstm_106_while_identity_5+
'lstm_106_while_lstm_106_strided_slice_1g
clstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensorM
;lstm_106_while_lstm_cell_111_matmul_readvariableop_resource:xO
=lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource:xJ
<lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource:x��3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp�2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp�4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp�
@lstm_106/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2lstm_106/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensor_0lstm_106_while_placeholderIlstm_106/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp=lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
#lstm_106/while/lstm_cell_111/MatMulMatMul9lstm_106/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp?lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
%lstm_106/while/lstm_cell_111/MatMul_1MatMullstm_106_while_placeholder_2<lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
 lstm_106/while/lstm_cell_111/addAddV2-lstm_106/while/lstm_cell_111/MatMul:product:0/lstm_106/while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp>lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
$lstm_106/while/lstm_cell_111/BiasAddBiasAdd$lstm_106/while/lstm_cell_111/add:z:0;lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xn
,lstm_106/while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
"lstm_106/while/lstm_cell_111/splitSplit5lstm_106/while/lstm_cell_111/split/split_dim:output:0-lstm_106/while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
$lstm_106/while/lstm_cell_111/SigmoidSigmoid+lstm_106/while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:����������
&lstm_106/while/lstm_cell_111/Sigmoid_1Sigmoid+lstm_106/while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
 lstm_106/while/lstm_cell_111/mulMul*lstm_106/while/lstm_cell_111/Sigmoid_1:y:0lstm_106_while_placeholder_3*
T0*'
_output_shapes
:����������
!lstm_106/while/lstm_cell_111/ReluRelu+lstm_106/while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
"lstm_106/while/lstm_cell_111/mul_1Mul(lstm_106/while/lstm_cell_111/Sigmoid:y:0/lstm_106/while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
"lstm_106/while/lstm_cell_111/add_1AddV2$lstm_106/while/lstm_cell_111/mul:z:0&lstm_106/while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:����������
&lstm_106/while/lstm_cell_111/Sigmoid_2Sigmoid+lstm_106/while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:����������
#lstm_106/while/lstm_cell_111/Relu_1Relu&lstm_106/while/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
"lstm_106/while/lstm_cell_111/mul_2Mul*lstm_106/while/lstm_cell_111/Sigmoid_2:y:01lstm_106/while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
3lstm_106/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_106_while_placeholder_1lstm_106_while_placeholder&lstm_106/while/lstm_cell_111/mul_2:z:0*
_output_shapes
: *
element_dtype0:���V
lstm_106/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_106/while/addAddV2lstm_106_while_placeholderlstm_106/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_106/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_106/while/add_1AddV2*lstm_106_while_lstm_106_while_loop_counterlstm_106/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_106/while/IdentityIdentitylstm_106/while/add_1:z:0^lstm_106/while/NoOp*
T0*
_output_shapes
: �
lstm_106/while/Identity_1Identity0lstm_106_while_lstm_106_while_maximum_iterations^lstm_106/while/NoOp*
T0*
_output_shapes
: t
lstm_106/while/Identity_2Identitylstm_106/while/add:z:0^lstm_106/while/NoOp*
T0*
_output_shapes
: �
lstm_106/while/Identity_3IdentityClstm_106/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_106/while/NoOp*
T0*
_output_shapes
: �
lstm_106/while/Identity_4Identity&lstm_106/while/lstm_cell_111/mul_2:z:0^lstm_106/while/NoOp*
T0*'
_output_shapes
:����������
lstm_106/while/Identity_5Identity&lstm_106/while/lstm_cell_111/add_1:z:0^lstm_106/while/NoOp*
T0*'
_output_shapes
:����������
lstm_106/while/NoOpNoOp4^lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp3^lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp5^lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_106_while_identity lstm_106/while/Identity:output:0"?
lstm_106_while_identity_1"lstm_106/while/Identity_1:output:0"?
lstm_106_while_identity_2"lstm_106/while/Identity_2:output:0"?
lstm_106_while_identity_3"lstm_106/while/Identity_3:output:0"?
lstm_106_while_identity_4"lstm_106/while/Identity_4:output:0"?
lstm_106_while_identity_5"lstm_106/while/Identity_5:output:0"T
'lstm_106_while_lstm_106_strided_slice_1)lstm_106_while_lstm_106_strided_slice_1_0"~
<lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource>lstm_106_while_lstm_cell_111_biasadd_readvariableop_resource_0"�
=lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource?lstm_106_while_lstm_cell_111_matmul_1_readvariableop_resource_0"|
;lstm_106_while_lstm_cell_111_matmul_readvariableop_resource=lstm_106_while_lstm_cell_111_matmul_readvariableop_resource_0"�
clstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensorelstm_106_while_tensorarrayv2read_tensorlistgetitem_lstm_106_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2j
3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp3lstm_106/while/lstm_cell_111/BiasAdd/ReadVariableOp2h
2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp2lstm_106/while/lstm_cell_111/MatMul/ReadVariableOp2l
4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp4lstm_106/while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
while_cond_23131412
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23131412___redundant_placeholder06
2while_while_cond_23131412___redundant_placeholder16
2while_while_cond_23131412___redundant_placeholder26
2while_while_cond_23131412___redundant_placeholder3
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131354

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
�
�
+__inference_lstm_105_layer_call_fn_23134042

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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23131994s
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
�9
�
while_body_23135635
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_112_matmul_readvariableop_resource_0:xH
6while_lstm_cell_112_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_112_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_112_matmul_readvariableop_resource:xF
4while_lstm_cell_112_matmul_1_readvariableop_resource:xA
3while_lstm_cell_112_biasadd_readvariableop_resource:x��*while/lstm_cell_112/BiasAdd/ReadVariableOp�)while/lstm_cell_112/MatMul/ReadVariableOp�+while/lstm_cell_112/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_112/addAddV2$while/lstm_cell_112/MatMul:product:0&while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_112/BiasAddBiasAddwhile/lstm_cell_112/add:z:02while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_112/splitSplit,while/lstm_cell_112/split/split_dim:output:0$while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_112/SigmoidSigmoid"while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_1Sigmoid"while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mulMul!while/lstm_cell_112/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_112/ReluRelu"while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_1Mulwhile/lstm_cell_112/Sigmoid:y:0&while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/add_1AddV2while/lstm_cell_112/mul:z:0while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_2Sigmoid"while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_112/Relu_1Reluwhile/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_2Mul!while/lstm_cell_112/Sigmoid_2:y:0(while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_112/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_112/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_112/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_112/BiasAdd/ReadVariableOp*^while/lstm_cell_112/MatMul/ReadVariableOp,^while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_112_biasadd_readvariableop_resource5while_lstm_cell_112_biasadd_readvariableop_resource_0"n
4while_lstm_cell_112_matmul_1_readvariableop_resource6while_lstm_cell_112_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_112_matmul_readvariableop_resource4while_lstm_cell_112_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_112/BiasAdd/ReadVariableOp*while/lstm_cell_112/BiasAdd/ReadVariableOp2V
)while/lstm_cell_112/MatMul/ReadVariableOp)while/lstm_cell_112/MatMul/ReadVariableOp2Z
+while/lstm_cell_112/MatMul_1/ReadVariableOp+while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135880

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
�
�
while_cond_23135779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23135779___redundant_placeholder06
2while_while_cond_23135779___redundant_placeholder16
2while_while_cond_23135779___redundant_placeholder26
2while_while_cond_23135779___redundant_placeholder3
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
while_body_23134541
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_110_matmul_readvariableop_resource_0:xH
6while_lstm_cell_110_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_110_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_110_matmul_readvariableop_resource:xF
4while_lstm_cell_110_matmul_1_readvariableop_resource:xA
3while_lstm_cell_110_biasadd_readvariableop_resource:x��*while/lstm_cell_110/BiasAdd/ReadVariableOp�)while/lstm_cell_110/MatMul/ReadVariableOp�+while/lstm_cell_110/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_110/addAddV2$while/lstm_cell_110/MatMul:product:0&while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_110/BiasAddBiasAddwhile/lstm_cell_110/add:z:02while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_110/splitSplit,while/lstm_cell_110/split/split_dim:output:0$while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_110/SigmoidSigmoid"while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_1Sigmoid"while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mulMul!while/lstm_cell_110/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_110/ReluRelu"while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_1Mulwhile/lstm_cell_110/Sigmoid:y:0&while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/add_1AddV2while/lstm_cell_110/mul:z:0while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_2Sigmoid"while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_110/Relu_1Reluwhile/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_2Mul!while/lstm_cell_110/Sigmoid_2:y:0(while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_110/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_110/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_110/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_110/BiasAdd/ReadVariableOp*^while/lstm_cell_110/MatMul/ReadVariableOp,^while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_110_biasadd_readvariableop_resource5while_lstm_cell_110_biasadd_readvariableop_resource_0"n
4while_lstm_cell_110_matmul_1_readvariableop_resource6while_lstm_cell_110_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_110_matmul_readvariableop_resource4while_lstm_cell_110_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_110/BiasAdd/ReadVariableOp*while/lstm_cell_110/BiasAdd/ReadVariableOp2V
)while/lstm_cell_110/MatMul/ReadVariableOp)while/lstm_cell_110/MatMul/ReadVariableOp2Z
+while/lstm_cell_110/MatMul_1/ReadVariableOp+while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
lstm_106_while_cond_23133339.
*lstm_106_while_lstm_106_while_loop_counter4
0lstm_106_while_lstm_106_while_maximum_iterations
lstm_106_while_placeholder 
lstm_106_while_placeholder_1 
lstm_106_while_placeholder_2 
lstm_106_while_placeholder_30
,lstm_106_while_less_lstm_106_strided_slice_1H
Dlstm_106_while_lstm_106_while_cond_23133339___redundant_placeholder0H
Dlstm_106_while_lstm_106_while_cond_23133339___redundant_placeholder1H
Dlstm_106_while_lstm_106_while_cond_23133339___redundant_placeholder2H
Dlstm_106_while_lstm_106_while_cond_23133339___redundant_placeholder3
lstm_106_while_identity
�
lstm_106/while/LessLesslstm_106_while_placeholder,lstm_106_while_less_lstm_106_strided_slice_1*
T0*
_output_shapes
: ]
lstm_106/while/IdentityIdentitylstm_106/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_106_while_identity lstm_106/while/Identity:output:0*(
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136075

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
�
�
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131558

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
�
�
while_cond_23131572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23131572___redundant_placeholder06
2while_while_cond_23131572___redundant_placeholder16
2while_while_cond_23131572___redundant_placeholder26
2while_while_cond_23131572___redundant_placeholder3
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

�
lstm_105_while_cond_23133200.
*lstm_105_while_lstm_105_while_loop_counter4
0lstm_105_while_lstm_105_while_maximum_iterations
lstm_105_while_placeholder 
lstm_105_while_placeholder_1 
lstm_105_while_placeholder_2 
lstm_105_while_placeholder_30
,lstm_105_while_less_lstm_105_strided_slice_1H
Dlstm_105_while_lstm_105_while_cond_23133200___redundant_placeholder0H
Dlstm_105_while_lstm_105_while_cond_23133200___redundant_placeholder1H
Dlstm_105_while_lstm_105_while_cond_23133200___redundant_placeholder2H
Dlstm_105_while_lstm_105_while_cond_23133200___redundant_placeholder3
lstm_105_while_identity
�
lstm_105/while/LessLesslstm_105_while_placeholder,lstm_105_while_less_lstm_105_strided_slice_1*
T0*
_output_shapes
: ]
lstm_105/while/IdentityIdentitylstm_105/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_105_while_identity lstm_105/while/Identity:output:0*(
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
while_body_23132790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_110_matmul_readvariableop_resource_0:xH
6while_lstm_cell_110_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_110_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_110_matmul_readvariableop_resource:xF
4while_lstm_cell_110_matmul_1_readvariableop_resource:xA
3while_lstm_cell_110_biasadd_readvariableop_resource:x��*while/lstm_cell_110/BiasAdd/ReadVariableOp�)while/lstm_cell_110/MatMul/ReadVariableOp�+while/lstm_cell_110/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_110/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_110_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_110_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_110/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_110/addAddV2$while/lstm_cell_110/MatMul:product:0&while/lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_110_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_110/BiasAddBiasAddwhile/lstm_cell_110/add:z:02while/lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_110/splitSplit,while/lstm_cell_110/split/split_dim:output:0$while/lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_110/SigmoidSigmoid"while/lstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_1Sigmoid"while/lstm_cell_110/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mulMul!while/lstm_cell_110/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_110/ReluRelu"while/lstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_1Mulwhile/lstm_cell_110/Sigmoid:y:0&while/lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/add_1AddV2while/lstm_cell_110/mul:z:0while/lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_110/Sigmoid_2Sigmoid"while/lstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_110/Relu_1Reluwhile/lstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_110/mul_2Mul!while/lstm_cell_110/Sigmoid_2:y:0(while/lstm_cell_110/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_110/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_110/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_110/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_110/BiasAdd/ReadVariableOp*^while/lstm_cell_110/MatMul/ReadVariableOp,^while/lstm_cell_110/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_110_biasadd_readvariableop_resource5while_lstm_cell_110_biasadd_readvariableop_resource_0"n
4while_lstm_cell_110_matmul_1_readvariableop_resource6while_lstm_cell_110_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_110_matmul_readvariableop_resource4while_lstm_cell_110_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_110/BiasAdd/ReadVariableOp*while/lstm_cell_110/BiasAdd/ReadVariableOp2V
)while/lstm_cell_110/MatMul/ReadVariableOp)while/lstm_cell_110/MatMul/ReadVariableOp2Z
+while/lstm_cell_110/MatMul_1/ReadVariableOp+while/lstm_cell_110/MatMul_1/ReadVariableOp: 
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23132328

inputs#
lstm_105_23131995:x#
lstm_105_23131997:x
lstm_105_23131999:x#
lstm_106_23132145:x#
lstm_106_23132147:x
lstm_106_23132149:x#
lstm_107_23132297:x#
lstm_107_23132299:x
lstm_107_23132301:x#
dense_85_23132322:
dense_85_23132324:
identity�� dense_85/StatefulPartitionedCall� lstm_105/StatefulPartitionedCall� lstm_106/StatefulPartitionedCall� lstm_107/StatefulPartitionedCall�
 lstm_105/StatefulPartitionedCallStatefulPartitionedCallinputslstm_105_23131995lstm_105_23131997lstm_105_23131999*
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23131994�
 lstm_106/StatefulPartitionedCallStatefulPartitionedCall)lstm_105/StatefulPartitionedCall:output:0lstm_106_23132145lstm_106_23132147lstm_106_23132149*
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132144�
 lstm_107/StatefulPartitionedCallStatefulPartitionedCall)lstm_106/StatefulPartitionedCall:output:0lstm_107_23132297lstm_107_23132299lstm_107_23132301*
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132296�
dropout_68/PartitionedCallPartitionedCall)lstm_107/StatefulPartitionedCall:output:0*
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132309�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall#dropout_68/PartitionedCall:output:0dense_85_23132322dense_85_23132324*
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
F__inference_dense_85_layer_call_and_return_conditional_losses_23132321x
IdentityIdentity)dense_85/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_85/StatefulPartitionedCall!^lstm_105/StatefulPartitionedCall!^lstm_106/StatefulPartitionedCall!^lstm_107/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 lstm_105/StatefulPartitionedCall lstm_105/StatefulPartitionedCall2D
 lstm_106/StatefulPartitionedCall lstm_106/StatefulPartitionedCall2D
 lstm_107/StatefulPartitionedCall lstm_107/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132309

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
�9
�
while_body_23135345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_112_matmul_readvariableop_resource_0:xH
6while_lstm_cell_112_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_112_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_112_matmul_readvariableop_resource:xF
4while_lstm_cell_112_matmul_1_readvariableop_resource:xA
3while_lstm_cell_112_biasadd_readvariableop_resource:x��*while/lstm_cell_112/BiasAdd/ReadVariableOp�)while/lstm_cell_112/MatMul/ReadVariableOp�+while/lstm_cell_112/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_112/addAddV2$while/lstm_cell_112/MatMul:product:0&while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_112/BiasAddBiasAddwhile/lstm_cell_112/add:z:02while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_112/splitSplit,while/lstm_cell_112/split/split_dim:output:0$while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_112/SigmoidSigmoid"while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_1Sigmoid"while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mulMul!while/lstm_cell_112/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_112/ReluRelu"while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_1Mulwhile/lstm_cell_112/Sigmoid:y:0&while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/add_1AddV2while/lstm_cell_112/mul:z:0while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_2Sigmoid"while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_112/Relu_1Reluwhile/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_2Mul!while/lstm_cell_112/Sigmoid_2:y:0(while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_112/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_112/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_112/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_112/BiasAdd/ReadVariableOp*^while/lstm_cell_112/MatMul/ReadVariableOp,^while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_112_biasadd_readvariableop_resource5while_lstm_cell_112_biasadd_readvariableop_resource_0"n
4while_lstm_cell_112_matmul_1_readvariableop_resource6while_lstm_cell_112_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_112_matmul_readvariableop_resource4while_lstm_cell_112_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_112/BiasAdd/ReadVariableOp*while/lstm_cell_112/BiasAdd/ReadVariableOp2V
)while/lstm_cell_112/MatMul/ReadVariableOp)while/lstm_cell_112/MatMul/ReadVariableOp2Z
+while/lstm_cell_112/MatMul_1/ReadVariableOp+while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
�J
�
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135098

inputs>
,lstm_cell_111_matmul_readvariableop_resource:x@
.lstm_cell_111_matmul_1_readvariableop_resource:x;
-lstm_cell_111_biasadd_readvariableop_resource:x
identity��$lstm_cell_111/BiasAdd/ReadVariableOp�#lstm_cell_111/MatMul/ReadVariableOp�%lstm_cell_111/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_111/MatMul/ReadVariableOpReadVariableOp,lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMulMatMulstrided_slice_2:output:0+lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMul_1MatMulzeros:output:0-lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_111/addAddV2lstm_cell_111/MatMul:product:0 lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_111/BiasAddBiasAddlstm_cell_111/add:z:0,lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_111/splitSplit&lstm_cell_111/split/split_dim:output:0lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_111/SigmoidSigmoidlstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_1Sigmoidlstm_cell_111/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_111/mulMullstm_cell_111/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_111/ReluRelulstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_1Mullstm_cell_111/Sigmoid:y:0 lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_111/add_1AddV2lstm_cell_111/mul:z:0lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_2Sigmoidlstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_111/Relu_1Relulstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_2Mullstm_cell_111/Sigmoid_2:y:0"lstm_cell_111/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_111_matmul_readvariableop_resource.lstm_cell_111_matmul_1_readvariableop_resource-lstm_cell_111_biasadd_readvariableop_resource*
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
while_body_23135014*
condR
while_cond_23135013*K
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
NoOpNoOp%^lstm_cell_111/BiasAdd/ReadVariableOp$^lstm_cell_111/MatMul/ReadVariableOp&^lstm_cell_111/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_111/BiasAdd/ReadVariableOp$lstm_cell_111/BiasAdd/ReadVariableOp2J
#lstm_cell_111/MatMul/ReadVariableOp#lstm_cell_111/MatMul/ReadVariableOp2N
%lstm_cell_111/MatMul_1/ReadVariableOp%lstm_cell_111/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134625

inputs>
,lstm_cell_110_matmul_readvariableop_resource:x@
.lstm_cell_110_matmul_1_readvariableop_resource:x;
-lstm_cell_110_biasadd_readvariableop_resource:x
identity��$lstm_cell_110/BiasAdd/ReadVariableOp�#lstm_cell_110/MatMul/ReadVariableOp�%lstm_cell_110/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_110/MatMul/ReadVariableOpReadVariableOp,lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMulMatMulstrided_slice_2:output:0+lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMul_1MatMulzeros:output:0-lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_110/addAddV2lstm_cell_110/MatMul:product:0 lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_110/BiasAddBiasAddlstm_cell_110/add:z:0,lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_110/splitSplit&lstm_cell_110/split/split_dim:output:0lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_110/SigmoidSigmoidlstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_1Sigmoidlstm_cell_110/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_110/mulMullstm_cell_110/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_110/ReluRelulstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_1Mullstm_cell_110/Sigmoid:y:0 lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_110/add_1AddV2lstm_cell_110/mul:z:0lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_2Sigmoidlstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_110/Relu_1Relulstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_2Mullstm_cell_110/Sigmoid_2:y:0"lstm_cell_110/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_110_matmul_readvariableop_resource.lstm_cell_110_matmul_1_readvariableop_resource-lstm_cell_110_biasadd_readvariableop_resource*
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
while_body_23134541*
condR
while_cond_23134540*K
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
NoOpNoOp%^lstm_cell_110/BiasAdd/ReadVariableOp$^lstm_cell_110/MatMul/ReadVariableOp&^lstm_cell_110/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_110/BiasAdd/ReadVariableOp$lstm_cell_110/BiasAdd/ReadVariableOp2J
#lstm_cell_110/MatMul/ReadVariableOp#lstm_cell_110/MatMul/ReadVariableOp2N
%lstm_cell_110/MatMul_1/ReadVariableOp%lstm_cell_110/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23135977

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
�K
�
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134955
inputs_0>
,lstm_cell_111_matmul_readvariableop_resource:x@
.lstm_cell_111_matmul_1_readvariableop_resource:x;
-lstm_cell_111_biasadd_readvariableop_resource:x
identity��$lstm_cell_111/BiasAdd/ReadVariableOp�#lstm_cell_111/MatMul/ReadVariableOp�%lstm_cell_111/MatMul_1/ReadVariableOp�while=
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
#lstm_cell_111/MatMul/ReadVariableOpReadVariableOp,lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMulMatMulstrided_slice_2:output:0+lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMul_1MatMulzeros:output:0-lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_111/addAddV2lstm_cell_111/MatMul:product:0 lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_111/BiasAddBiasAddlstm_cell_111/add:z:0,lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_111/splitSplit&lstm_cell_111/split/split_dim:output:0lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_111/SigmoidSigmoidlstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_1Sigmoidlstm_cell_111/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_111/mulMullstm_cell_111/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_111/ReluRelulstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_1Mullstm_cell_111/Sigmoid:y:0 lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_111/add_1AddV2lstm_cell_111/mul:z:0lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_2Sigmoidlstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_111/Relu_1Relulstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_2Mullstm_cell_111/Sigmoid_2:y:0"lstm_cell_111/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_111_matmul_readvariableop_resource.lstm_cell_111_matmul_1_readvariableop_resource-lstm_cell_111_biasadd_readvariableop_resource*
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
while_body_23134871*
condR
while_cond_23134870*K
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
NoOpNoOp%^lstm_cell_111/BiasAdd/ReadVariableOp$^lstm_cell_111/MatMul/ReadVariableOp&^lstm_cell_111/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_111/BiasAdd/ReadVariableOp$lstm_cell_111/BiasAdd/ReadVariableOp2J
#lstm_cell_111/MatMul/ReadVariableOp#lstm_cell_111/MatMul/ReadVariableOp2N
%lstm_cell_111/MatMul_1/ReadVariableOp%lstm_cell_111/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
0__inference_lstm_cell_110_layer_call_fn_23135945

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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23131004o
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
while_cond_23135634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23135634___redundant_placeholder06
2while_while_cond_23135634___redundant_placeholder16
2while_while_cond_23135634___redundant_placeholder26
2while_while_cond_23135634___redundant_placeholder3
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
while_body_23132625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_111_matmul_readvariableop_resource_0:xH
6while_lstm_cell_111_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_111_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_111_matmul_readvariableop_resource:xF
4while_lstm_cell_111_matmul_1_readvariableop_resource:xA
3while_lstm_cell_111_biasadd_readvariableop_resource:x��*while/lstm_cell_111/BiasAdd/ReadVariableOp�)while/lstm_cell_111/MatMul/ReadVariableOp�+while/lstm_cell_111/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_111/addAddV2$while/lstm_cell_111/MatMul:product:0&while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_111/BiasAddBiasAddwhile/lstm_cell_111/add:z:02while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_111/splitSplit,while/lstm_cell_111/split/split_dim:output:0$while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_111/SigmoidSigmoid"while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_1Sigmoid"while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mulMul!while/lstm_cell_111/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_111/ReluRelu"while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_1Mulwhile/lstm_cell_111/Sigmoid:y:0&while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/add_1AddV2while/lstm_cell_111/mul:z:0while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_2Sigmoid"while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_111/Relu_1Reluwhile/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_2Mul!while/lstm_cell_111/Sigmoid_2:y:0(while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_111/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_111/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_111/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_111/BiasAdd/ReadVariableOp*^while/lstm_cell_111/MatMul/ReadVariableOp,^while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_111_biasadd_readvariableop_resource5while_lstm_cell_111_biasadd_readvariableop_resource_0"n
4while_lstm_cell_111_matmul_1_readvariableop_resource6while_lstm_cell_111_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_111_matmul_readvariableop_resource4while_lstm_cell_111_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_111/BiasAdd/ReadVariableOp*while/lstm_cell_111/BiasAdd/ReadVariableOp2V
)while/lstm_cell_111/MatMul/ReadVariableOp)while/lstm_cell_111/MatMul/ReadVariableOp2Z
+while/lstm_cell_111/MatMul_1/ReadVariableOp+while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132296

inputs>
,lstm_cell_112_matmul_readvariableop_resource:x@
.lstm_cell_112_matmul_1_readvariableop_resource:x;
-lstm_cell_112_biasadd_readvariableop_resource:x
identity��$lstm_cell_112/BiasAdd/ReadVariableOp�#lstm_cell_112/MatMul/ReadVariableOp�%lstm_cell_112/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_112/MatMul/ReadVariableOpReadVariableOp,lstm_cell_112_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMulMatMulstrided_slice_2:output:0+lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_112_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_112/MatMul_1MatMulzeros:output:0-lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_112/addAddV2lstm_cell_112/MatMul:product:0 lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_112_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_112/BiasAddBiasAddlstm_cell_112/add:z:0,lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_112/splitSplit&lstm_cell_112/split/split_dim:output:0lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_112/SigmoidSigmoidlstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_1Sigmoidlstm_cell_112/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_112/mulMullstm_cell_112/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_112/ReluRelulstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_1Mullstm_cell_112/Sigmoid:y:0 lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_112/add_1AddV2lstm_cell_112/mul:z:0lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_112/Sigmoid_2Sigmoidlstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_112/Relu_1Relulstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_112/mul_2Mullstm_cell_112/Sigmoid_2:y:0"lstm_cell_112/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_112_matmul_readvariableop_resource.lstm_cell_112_matmul_1_readvariableop_resource-lstm_cell_112_biasadd_readvariableop_resource*
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
while_body_23132211*
condR
while_cond_23132210*K
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
NoOpNoOp%^lstm_cell_112/BiasAdd/ReadVariableOp$^lstm_cell_112/MatMul/ReadVariableOp&^lstm_cell_112/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_112/BiasAdd/ReadVariableOp$lstm_cell_112/BiasAdd/ReadVariableOp2J
#lstm_cell_112/MatMul/ReadVariableOp#lstm_cell_112/MatMul/ReadVariableOp2N
%lstm_cell_112/MatMul_1/ReadVariableOp%lstm_cell_112/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
while_body_23135780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_112_matmul_readvariableop_resource_0:xH
6while_lstm_cell_112_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_112_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_112_matmul_readvariableop_resource:xF
4while_lstm_cell_112_matmul_1_readvariableop_resource:xA
3while_lstm_cell_112_biasadd_readvariableop_resource:x��*while/lstm_cell_112/BiasAdd/ReadVariableOp�)while/lstm_cell_112/MatMul/ReadVariableOp�+while/lstm_cell_112/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_112/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_112/addAddV2$while/lstm_cell_112/MatMul:product:0&while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_112/BiasAddBiasAddwhile/lstm_cell_112/add:z:02while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_112/splitSplit,while/lstm_cell_112/split/split_dim:output:0$while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_112/SigmoidSigmoid"while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_1Sigmoid"while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mulMul!while/lstm_cell_112/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_112/ReluRelu"while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_1Mulwhile/lstm_cell_112/Sigmoid:y:0&while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/add_1AddV2while/lstm_cell_112/mul:z:0while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_112/Sigmoid_2Sigmoid"while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_112/Relu_1Reluwhile/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_112/mul_2Mul!while/lstm_cell_112/Sigmoid_2:y:0(while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:���������r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_112/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_112/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_112/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_112/BiasAdd/ReadVariableOp*^while/lstm_cell_112/MatMul/ReadVariableOp,^while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_112_biasadd_readvariableop_resource5while_lstm_cell_112_biasadd_readvariableop_resource_0"n
4while_lstm_cell_112_matmul_1_readvariableop_resource6while_lstm_cell_112_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_112_matmul_readvariableop_resource4while_lstm_cell_112_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_112/BiasAdd/ReadVariableOp*while/lstm_cell_112/BiasAdd/ReadVariableOp2V
)while/lstm_cell_112/MatMul/ReadVariableOp)while/lstm_cell_112/MatMul/ReadVariableOp2Z
+while/lstm_cell_112/MatMul_1/ReadVariableOp+while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23131643

inputs(
lstm_cell_112_23131559:x(
lstm_cell_112_23131561:x$
lstm_cell_112_23131563:x
identity��%lstm_cell_112/StatefulPartitionedCall�while;
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
%lstm_cell_112/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_112_23131559lstm_cell_112_23131561lstm_cell_112_23131563*
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131558n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_112_23131559lstm_cell_112_23131561lstm_cell_112_23131563*
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
while_body_23131573*
condR
while_cond_23131572*K
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
NoOpNoOp&^lstm_cell_112/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2N
%lstm_cell_112/StatefulPartitionedCall%lstm_cell_112/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133057
lstm_105_input#
lstm_105_23133029:x#
lstm_105_23133031:x
lstm_105_23133033:x#
lstm_106_23133036:x#
lstm_106_23133038:x
lstm_106_23133040:x#
lstm_107_23133043:x#
lstm_107_23133045:x
lstm_107_23133047:x#
dense_85_23133051:
dense_85_23133053:
identity�� dense_85/StatefulPartitionedCall�"dropout_68/StatefulPartitionedCall� lstm_105/StatefulPartitionedCall� lstm_106/StatefulPartitionedCall� lstm_107/StatefulPartitionedCall�
 lstm_105/StatefulPartitionedCallStatefulPartitionedCalllstm_105_inputlstm_105_23133029lstm_105_23133031lstm_105_23133033*
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23132874�
 lstm_106/StatefulPartitionedCallStatefulPartitionedCall)lstm_105/StatefulPartitionedCall:output:0lstm_106_23133036lstm_106_23133038lstm_106_23133040*
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23132709�
 lstm_107/StatefulPartitionedCallStatefulPartitionedCall)lstm_106/StatefulPartitionedCall:output:0lstm_107_23133043lstm_107_23133045lstm_107_23133047*
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23132544�
"dropout_68/StatefulPartitionedCallStatefulPartitionedCall)lstm_107/StatefulPartitionedCall:output:0*
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132383�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall+dropout_68/StatefulPartitionedCall:output:0dense_85_23133051dense_85_23133053*
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
F__inference_dense_85_layer_call_and_return_conditional_losses_23132321x
IdentityIdentity)dense_85/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_85/StatefulPartitionedCall#^dropout_68/StatefulPartitionedCall!^lstm_105/StatefulPartitionedCall!^lstm_106/StatefulPartitionedCall!^lstm_107/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : 2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2H
"dropout_68/StatefulPartitionedCall"dropout_68/StatefulPartitionedCall2D
 lstm_105/StatefulPartitionedCall lstm_105/StatefulPartitionedCall2D
 lstm_106/StatefulPartitionedCall lstm_106/StatefulPartitionedCall2D
 lstm_107/StatefulPartitionedCall lstm_107/StatefulPartitionedCall:[ W
+
_output_shapes
:���������
(
_user_specified_namelstm_105_input
�
�
while_cond_23131765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23131765___redundant_placeholder06
2while_while_cond_23131765___redundant_placeholder16
2while_while_cond_23131765___redundant_placeholder26
2while_while_cond_23131765___redundant_placeholder3
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
�
f
-__inference_dropout_68_layer_call_fn_23135875

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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132383o
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
�U
�
*sequential_87_lstm_107_while_body_23130699J
Fsequential_87_lstm_107_while_sequential_87_lstm_107_while_loop_counterP
Lsequential_87_lstm_107_while_sequential_87_lstm_107_while_maximum_iterations,
(sequential_87_lstm_107_while_placeholder.
*sequential_87_lstm_107_while_placeholder_1.
*sequential_87_lstm_107_while_placeholder_2.
*sequential_87_lstm_107_while_placeholder_3I
Esequential_87_lstm_107_while_sequential_87_lstm_107_strided_slice_1_0�
�sequential_87_lstm_107_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_107_tensorarrayunstack_tensorlistfromtensor_0]
Ksequential_87_lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0:x_
Msequential_87_lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0:xZ
Lsequential_87_lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0:x)
%sequential_87_lstm_107_while_identity+
'sequential_87_lstm_107_while_identity_1+
'sequential_87_lstm_107_while_identity_2+
'sequential_87_lstm_107_while_identity_3+
'sequential_87_lstm_107_while_identity_4+
'sequential_87_lstm_107_while_identity_5G
Csequential_87_lstm_107_while_sequential_87_lstm_107_strided_slice_1�
sequential_87_lstm_107_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_107_tensorarrayunstack_tensorlistfromtensor[
Isequential_87_lstm_107_while_lstm_cell_112_matmul_readvariableop_resource:x]
Ksequential_87_lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource:xX
Jsequential_87_lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource:x��Asequential_87/lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp�@sequential_87/lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp�Bsequential_87/lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp�
Nsequential_87/lstm_107/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
@sequential_87/lstm_107/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_87_lstm_107_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_107_tensorarrayunstack_tensorlistfromtensor_0(sequential_87_lstm_107_while_placeholderWsequential_87/lstm_107/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
@sequential_87/lstm_107/while/lstm_cell_112/MatMul/ReadVariableOpReadVariableOpKsequential_87_lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
1sequential_87/lstm_107/while/lstm_cell_112/MatMulMatMulGsequential_87/lstm_107/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_87/lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
Bsequential_87/lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOpReadVariableOpMsequential_87_lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
3sequential_87/lstm_107/while/lstm_cell_112/MatMul_1MatMul*sequential_87_lstm_107_while_placeholder_2Jsequential_87/lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
.sequential_87/lstm_107/while/lstm_cell_112/addAddV2;sequential_87/lstm_107/while/lstm_cell_112/MatMul:product:0=sequential_87/lstm_107/while/lstm_cell_112/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
Asequential_87/lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOpReadVariableOpLsequential_87_lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
2sequential_87/lstm_107/while/lstm_cell_112/BiasAddBiasAdd2sequential_87/lstm_107/while/lstm_cell_112/add:z:0Isequential_87/lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x|
:sequential_87/lstm_107/while/lstm_cell_112/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
0sequential_87/lstm_107/while/lstm_cell_112/splitSplitCsequential_87/lstm_107/while/lstm_cell_112/split/split_dim:output:0;sequential_87/lstm_107/while/lstm_cell_112/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
2sequential_87/lstm_107/while/lstm_cell_112/SigmoidSigmoid9sequential_87/lstm_107/while/lstm_cell_112/split:output:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_107/while/lstm_cell_112/Sigmoid_1Sigmoid9sequential_87/lstm_107/while/lstm_cell_112/split:output:1*
T0*'
_output_shapes
:����������
.sequential_87/lstm_107/while/lstm_cell_112/mulMul8sequential_87/lstm_107/while/lstm_cell_112/Sigmoid_1:y:0*sequential_87_lstm_107_while_placeholder_3*
T0*'
_output_shapes
:����������
/sequential_87/lstm_107/while/lstm_cell_112/ReluRelu9sequential_87/lstm_107/while/lstm_cell_112/split:output:2*
T0*'
_output_shapes
:����������
0sequential_87/lstm_107/while/lstm_cell_112/mul_1Mul6sequential_87/lstm_107/while/lstm_cell_112/Sigmoid:y:0=sequential_87/lstm_107/while/lstm_cell_112/Relu:activations:0*
T0*'
_output_shapes
:����������
0sequential_87/lstm_107/while/lstm_cell_112/add_1AddV22sequential_87/lstm_107/while/lstm_cell_112/mul:z:04sequential_87/lstm_107/while/lstm_cell_112/mul_1:z:0*
T0*'
_output_shapes
:����������
4sequential_87/lstm_107/while/lstm_cell_112/Sigmoid_2Sigmoid9sequential_87/lstm_107/while/lstm_cell_112/split:output:3*
T0*'
_output_shapes
:����������
1sequential_87/lstm_107/while/lstm_cell_112/Relu_1Relu4sequential_87/lstm_107/while/lstm_cell_112/add_1:z:0*
T0*'
_output_shapes
:����������
0sequential_87/lstm_107/while/lstm_cell_112/mul_2Mul8sequential_87/lstm_107/while/lstm_cell_112/Sigmoid_2:y:0?sequential_87/lstm_107/while/lstm_cell_112/Relu_1:activations:0*
T0*'
_output_shapes
:����������
Gsequential_87/lstm_107/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
Asequential_87/lstm_107/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_87_lstm_107_while_placeholder_1Psequential_87/lstm_107/while/TensorArrayV2Write/TensorListSetItem/index:output:04sequential_87/lstm_107/while/lstm_cell_112/mul_2:z:0*
_output_shapes
: *
element_dtype0:���d
"sequential_87/lstm_107/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
 sequential_87/lstm_107/while/addAddV2(sequential_87_lstm_107_while_placeholder+sequential_87/lstm_107/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_87/lstm_107/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
"sequential_87/lstm_107/while/add_1AddV2Fsequential_87_lstm_107_while_sequential_87_lstm_107_while_loop_counter-sequential_87/lstm_107/while/add_1/y:output:0*
T0*
_output_shapes
: �
%sequential_87/lstm_107/while/IdentityIdentity&sequential_87/lstm_107/while/add_1:z:0"^sequential_87/lstm_107/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_107/while/Identity_1IdentityLsequential_87_lstm_107_while_sequential_87_lstm_107_while_maximum_iterations"^sequential_87/lstm_107/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_107/while/Identity_2Identity$sequential_87/lstm_107/while/add:z:0"^sequential_87/lstm_107/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_107/while/Identity_3IdentityQsequential_87/lstm_107/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_87/lstm_107/while/NoOp*
T0*
_output_shapes
: �
'sequential_87/lstm_107/while/Identity_4Identity4sequential_87/lstm_107/while/lstm_cell_112/mul_2:z:0"^sequential_87/lstm_107/while/NoOp*
T0*'
_output_shapes
:����������
'sequential_87/lstm_107/while/Identity_5Identity4sequential_87/lstm_107/while/lstm_cell_112/add_1:z:0"^sequential_87/lstm_107/while/NoOp*
T0*'
_output_shapes
:����������
!sequential_87/lstm_107/while/NoOpNoOpB^sequential_87/lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOpA^sequential_87/lstm_107/while/lstm_cell_112/MatMul/ReadVariableOpC^sequential_87/lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_87_lstm_107_while_identity.sequential_87/lstm_107/while/Identity:output:0"[
'sequential_87_lstm_107_while_identity_10sequential_87/lstm_107/while/Identity_1:output:0"[
'sequential_87_lstm_107_while_identity_20sequential_87/lstm_107/while/Identity_2:output:0"[
'sequential_87_lstm_107_while_identity_30sequential_87/lstm_107/while/Identity_3:output:0"[
'sequential_87_lstm_107_while_identity_40sequential_87/lstm_107/while/Identity_4:output:0"[
'sequential_87_lstm_107_while_identity_50sequential_87/lstm_107/while/Identity_5:output:0"�
Jsequential_87_lstm_107_while_lstm_cell_112_biasadd_readvariableop_resourceLsequential_87_lstm_107_while_lstm_cell_112_biasadd_readvariableop_resource_0"�
Ksequential_87_lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resourceMsequential_87_lstm_107_while_lstm_cell_112_matmul_1_readvariableop_resource_0"�
Isequential_87_lstm_107_while_lstm_cell_112_matmul_readvariableop_resourceKsequential_87_lstm_107_while_lstm_cell_112_matmul_readvariableop_resource_0"�
Csequential_87_lstm_107_while_sequential_87_lstm_107_strided_slice_1Esequential_87_lstm_107_while_sequential_87_lstm_107_strided_slice_1_0"�
sequential_87_lstm_107_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_107_tensorarrayunstack_tensorlistfromtensor�sequential_87_lstm_107_while_tensorarrayv2read_tensorlistgetitem_sequential_87_lstm_107_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2�
Asequential_87/lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOpAsequential_87/lstm_107/while/lstm_cell_112/BiasAdd/ReadVariableOp2�
@sequential_87/lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp@sequential_87/lstm_107/while/lstm_cell_112/MatMul/ReadVariableOp2�
Bsequential_87/lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOpBsequential_87/lstm_107/while/lstm_cell_112/MatMul_1/ReadVariableOp: 
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
while_body_23135014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
4while_lstm_cell_111_matmul_readvariableop_resource_0:xH
6while_lstm_cell_111_matmul_1_readvariableop_resource_0:xC
5while_lstm_cell_111_biasadd_readvariableop_resource_0:x
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
2while_lstm_cell_111_matmul_readvariableop_resource:xF
4while_lstm_cell_111_matmul_1_readvariableop_resource:xA
3while_lstm_cell_111_biasadd_readvariableop_resource:x��*while/lstm_cell_111/BiasAdd/ReadVariableOp�)while/lstm_cell_111/MatMul/ReadVariableOp�+while/lstm_cell_111/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
)while/lstm_cell_111/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_111_matmul_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
+while/lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_111_matmul_1_readvariableop_resource_0*
_output_shapes

:x*
dtype0�
while/lstm_cell_111/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
while/lstm_cell_111/addAddV2$while/lstm_cell_111/MatMul:product:0&while/lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
*while/lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_111_biasadd_readvariableop_resource_0*
_output_shapes
:x*
dtype0�
while/lstm_cell_111/BiasAddBiasAddwhile/lstm_cell_111/add:z:02while/lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xe
#while/lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell_111/splitSplit,while/lstm_cell_111/split/split_dim:output:0$while/lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split|
while/lstm_cell_111/SigmoidSigmoid"while/lstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_1Sigmoid"while/lstm_cell_111/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mulMul!while/lstm_cell_111/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������v
while/lstm_cell_111/ReluRelu"while/lstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_1Mulwhile/lstm_cell_111/Sigmoid:y:0&while/lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/add_1AddV2while/lstm_cell_111/mul:z:0while/lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������~
while/lstm_cell_111/Sigmoid_2Sigmoid"while/lstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������s
while/lstm_cell_111/Relu_1Reluwhile/lstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell_111/mul_2Mul!while/lstm_cell_111/Sigmoid_2:y:0(while/lstm_cell_111/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_111/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_111/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������z
while/Identity_5Identitywhile/lstm_cell_111/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp+^while/lstm_cell_111/BiasAdd/ReadVariableOp*^while/lstm_cell_111/MatMul/ReadVariableOp,^while/lstm_cell_111/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_111_biasadd_readvariableop_resource5while_lstm_cell_111_biasadd_readvariableop_resource_0"n
4while_lstm_cell_111_matmul_1_readvariableop_resource6while_lstm_cell_111_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_111_matmul_readvariableop_resource4while_lstm_cell_111_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2X
*while/lstm_cell_111/BiasAdd/ReadVariableOp*while/lstm_cell_111/BiasAdd/ReadVariableOp2V
)while/lstm_cell_111/MatMul/ReadVariableOp)while/lstm_cell_111/MatMul/ReadVariableOp2Z
+while/lstm_cell_111/MatMul_1/ReadVariableOp+while/lstm_cell_111/MatMul_1/ReadVariableOp: 
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
0__inference_lstm_cell_112_layer_call_fn_23136141

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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131706o
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
�
�
while_cond_23134254
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23134254___redundant_placeholder06
2while_while_cond_23134254___redundant_placeholder16
2while_while_cond_23134254___redundant_placeholder26
2while_while_cond_23134254___redundant_placeholder3
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23132874

inputs>
,lstm_cell_110_matmul_readvariableop_resource:x@
.lstm_cell_110_matmul_1_readvariableop_resource:x;
-lstm_cell_110_biasadd_readvariableop_resource:x
identity��$lstm_cell_110/BiasAdd/ReadVariableOp�#lstm_cell_110/MatMul/ReadVariableOp�%lstm_cell_110/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_110/MatMul/ReadVariableOpReadVariableOp,lstm_cell_110_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMulMatMulstrided_slice_2:output:0+lstm_cell_110/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_110/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_110_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_110/MatMul_1MatMulzeros:output:0-lstm_cell_110/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_110/addAddV2lstm_cell_110/MatMul:product:0 lstm_cell_110/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_110/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_110_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_110/BiasAddBiasAddlstm_cell_110/add:z:0,lstm_cell_110/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_110/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_110/splitSplit&lstm_cell_110/split/split_dim:output:0lstm_cell_110/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_110/SigmoidSigmoidlstm_cell_110/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_1Sigmoidlstm_cell_110/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_110/mulMullstm_cell_110/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_110/ReluRelulstm_cell_110/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_1Mullstm_cell_110/Sigmoid:y:0 lstm_cell_110/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_110/add_1AddV2lstm_cell_110/mul:z:0lstm_cell_110/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_110/Sigmoid_2Sigmoidlstm_cell_110/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_110/Relu_1Relulstm_cell_110/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_110/mul_2Mullstm_cell_110/Sigmoid_2:y:0"lstm_cell_110/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_110_matmul_readvariableop_resource.lstm_cell_110_matmul_1_readvariableop_resource-lstm_cell_110_biasadd_readvariableop_resource*
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
while_body_23132790*
condR
while_cond_23132789*K
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
NoOpNoOp%^lstm_cell_110/BiasAdd/ReadVariableOp$^lstm_cell_110/MatMul/ReadVariableOp&^lstm_cell_110/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_110/BiasAdd/ReadVariableOp$lstm_cell_110/BiasAdd/ReadVariableOp2J
#lstm_cell_110/MatMul/ReadVariableOp#lstm_cell_110/MatMul/ReadVariableOp2N
%lstm_cell_110/MatMul_1/ReadVariableOp%lstm_cell_110/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136205

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
0__inference_lstm_cell_112_layer_call_fn_23136124

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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131558o
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23131706

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
�
0__inference_lstm_cell_111_layer_call_fn_23136026

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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23131208o
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
�
0__inference_lstm_cell_110_layer_call_fn_23135928

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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23130858o
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
while_cond_23135013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23135013___redundant_placeholder06
2while_while_cond_23135013___redundant_placeholder16
2while_while_cond_23135013___redundant_placeholder26
2while_while_cond_23135013___redundant_placeholder3
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135241

inputs>
,lstm_cell_111_matmul_readvariableop_resource:x@
.lstm_cell_111_matmul_1_readvariableop_resource:x;
-lstm_cell_111_biasadd_readvariableop_resource:x
identity��$lstm_cell_111/BiasAdd/ReadVariableOp�#lstm_cell_111/MatMul/ReadVariableOp�%lstm_cell_111/MatMul_1/ReadVariableOp�while;
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
#lstm_cell_111/MatMul/ReadVariableOpReadVariableOp,lstm_cell_111_matmul_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMulMatMulstrided_slice_2:output:0+lstm_cell_111/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
%lstm_cell_111/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_111_matmul_1_readvariableop_resource*
_output_shapes

:x*
dtype0�
lstm_cell_111/MatMul_1MatMulzeros:output:0-lstm_cell_111/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
lstm_cell_111/addAddV2lstm_cell_111/MatMul:product:0 lstm_cell_111/MatMul_1:product:0*
T0*'
_output_shapes
:���������x�
$lstm_cell_111/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_111_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
lstm_cell_111/BiasAddBiasAddlstm_cell_111/add:z:0,lstm_cell_111/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x_
lstm_cell_111/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell_111/splitSplit&lstm_cell_111/split/split_dim:output:0lstm_cell_111/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitp
lstm_cell_111/SigmoidSigmoidlstm_cell_111/split:output:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_1Sigmoidlstm_cell_111/split:output:1*
T0*'
_output_shapes
:���������y
lstm_cell_111/mulMullstm_cell_111/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������j
lstm_cell_111/ReluRelulstm_cell_111/split:output:2*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_1Mullstm_cell_111/Sigmoid:y:0 lstm_cell_111/Relu:activations:0*
T0*'
_output_shapes
:���������~
lstm_cell_111/add_1AddV2lstm_cell_111/mul:z:0lstm_cell_111/mul_1:z:0*
T0*'
_output_shapes
:���������r
lstm_cell_111/Sigmoid_2Sigmoidlstm_cell_111/split:output:3*
T0*'
_output_shapes
:���������g
lstm_cell_111/Relu_1Relulstm_cell_111/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell_111/mul_2Mullstm_cell_111/Sigmoid_2:y:0"lstm_cell_111/Relu_1:activations:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_111_matmul_readvariableop_resource.lstm_cell_111_matmul_1_readvariableop_resource-lstm_cell_111_biasadd_readvariableop_resource*
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
while_body_23135157*
condR
while_cond_23135156*K
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
NoOpNoOp%^lstm_cell_111/BiasAdd/ReadVariableOp$^lstm_cell_111/MatMul/ReadVariableOp&^lstm_cell_111/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2L
$lstm_cell_111/BiasAdd/ReadVariableOp$lstm_cell_111/BiasAdd/ReadVariableOp2J
#lstm_cell_111/MatMul/ReadVariableOp#lstm_cell_111/MatMul/ReadVariableOp2N
%lstm_cell_111/MatMul_1/ReadVariableOp%lstm_cell_111/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_dropout_68_layer_call_fn_23135870

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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23132309`
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
�
�
*sequential_87_lstm_107_while_cond_23130698J
Fsequential_87_lstm_107_while_sequential_87_lstm_107_while_loop_counterP
Lsequential_87_lstm_107_while_sequential_87_lstm_107_while_maximum_iterations,
(sequential_87_lstm_107_while_placeholder.
*sequential_87_lstm_107_while_placeholder_1.
*sequential_87_lstm_107_while_placeholder_2.
*sequential_87_lstm_107_while_placeholder_3L
Hsequential_87_lstm_107_while_less_sequential_87_lstm_107_strided_slice_1d
`sequential_87_lstm_107_while_sequential_87_lstm_107_while_cond_23130698___redundant_placeholder0d
`sequential_87_lstm_107_while_sequential_87_lstm_107_while_cond_23130698___redundant_placeholder1d
`sequential_87_lstm_107_while_sequential_87_lstm_107_while_cond_23130698___redundant_placeholder2d
`sequential_87_lstm_107_while_sequential_87_lstm_107_while_cond_23130698___redundant_placeholder3)
%sequential_87_lstm_107_while_identity
�
!sequential_87/lstm_107/while/LessLess(sequential_87_lstm_107_while_placeholderHsequential_87_lstm_107_while_less_sequential_87_lstm_107_strided_slice_1*
T0*
_output_shapes
: y
%sequential_87/lstm_107/while/IdentityIdentity%sequential_87/lstm_107/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_87_lstm_107_while_identity.sequential_87/lstm_107/while/Identity:output:0*(
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
lstm_105_input;
 serving_default_lstm_105_input:0���������<
dense_850
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
0__inference_sequential_87_layer_call_fn_23132353
0__inference_sequential_87_layer_call_fn_23133115
0__inference_sequential_87_layer_call_fn_23133142
0__inference_sequential_87_layer_call_fn_23132995�
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133572
K__inference_sequential_87_layer_call_and_return_conditional_losses_23134009
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133026
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133057�
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
#__inference__wrapped_model_23130791lstm_105_input"�
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
+__inference_lstm_105_layer_call_fn_23134020
+__inference_lstm_105_layer_call_fn_23134031
+__inference_lstm_105_layer_call_fn_23134042
+__inference_lstm_105_layer_call_fn_23134053�
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134196
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134339
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134482
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134625�
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
+__inference_lstm_106_layer_call_fn_23134636
+__inference_lstm_106_layer_call_fn_23134647
+__inference_lstm_106_layer_call_fn_23134658
+__inference_lstm_106_layer_call_fn_23134669�
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134812
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134955
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135098
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135241�
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
+__inference_lstm_107_layer_call_fn_23135252
+__inference_lstm_107_layer_call_fn_23135263
+__inference_lstm_107_layer_call_fn_23135274
+__inference_lstm_107_layer_call_fn_23135285�
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135430
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135575
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135720
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135865�
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
-__inference_dropout_68_layer_call_fn_23135870
-__inference_dropout_68_layer_call_fn_23135875�
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135880
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135892�
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
+__inference_dense_85_layer_call_fn_23135901�
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
F__inference_dense_85_layer_call_and_return_conditional_losses_23135911�
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
!:2dense_85/kernel
:2dense_85/bias
/:-x2lstm_105/lstm_cell_110/kernel
9:7x2'lstm_105/lstm_cell_110/recurrent_kernel
):'x2lstm_105/lstm_cell_110/bias
/:-x2lstm_106/lstm_cell_111/kernel
9:7x2'lstm_106/lstm_cell_111/recurrent_kernel
):'x2lstm_106/lstm_cell_111/bias
/:-x2lstm_107/lstm_cell_112/kernel
9:7x2'lstm_107/lstm_cell_112/recurrent_kernel
):'x2lstm_107/lstm_cell_112/bias
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
0__inference_sequential_87_layer_call_fn_23132353lstm_105_input"�
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
0__inference_sequential_87_layer_call_fn_23133115inputs"�
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
0__inference_sequential_87_layer_call_fn_23133142inputs"�
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
0__inference_sequential_87_layer_call_fn_23132995lstm_105_input"�
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133572inputs"�
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23134009inputs"�
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133026lstm_105_input"�
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133057lstm_105_input"�
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
&__inference_signature_wrapper_23133088lstm_105_input"�
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
+__inference_lstm_105_layer_call_fn_23134020inputs_0"�
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
+__inference_lstm_105_layer_call_fn_23134031inputs_0"�
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
+__inference_lstm_105_layer_call_fn_23134042inputs"�
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
+__inference_lstm_105_layer_call_fn_23134053inputs"�
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134196inputs_0"�
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134339inputs_0"�
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134482inputs"�
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134625inputs"�
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
0__inference_lstm_cell_110_layer_call_fn_23135928
0__inference_lstm_cell_110_layer_call_fn_23135945�
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23135977
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23136009�
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
+__inference_lstm_106_layer_call_fn_23134636inputs_0"�
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
+__inference_lstm_106_layer_call_fn_23134647inputs_0"�
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
+__inference_lstm_106_layer_call_fn_23134658inputs"�
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
+__inference_lstm_106_layer_call_fn_23134669inputs"�
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134812inputs_0"�
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134955inputs_0"�
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135098inputs"�
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135241inputs"�
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
0__inference_lstm_cell_111_layer_call_fn_23136026
0__inference_lstm_cell_111_layer_call_fn_23136043�
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136075
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136107�
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
+__inference_lstm_107_layer_call_fn_23135252inputs_0"�
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
+__inference_lstm_107_layer_call_fn_23135263inputs_0"�
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
+__inference_lstm_107_layer_call_fn_23135274inputs"�
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
+__inference_lstm_107_layer_call_fn_23135285inputs"�
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135430inputs_0"�
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135575inputs_0"�
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135720inputs"�
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135865inputs"�
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
0__inference_lstm_cell_112_layer_call_fn_23136124
0__inference_lstm_cell_112_layer_call_fn_23136141�
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136173
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136205�
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
-__inference_dropout_68_layer_call_fn_23135870inputs"�
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
-__inference_dropout_68_layer_call_fn_23135875inputs"�
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135880inputs"�
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
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135892inputs"�
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
+__inference_dense_85_layer_call_fn_23135901inputs"�
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
F__inference_dense_85_layer_call_and_return_conditional_losses_23135911inputs"�
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
4:2x2$Adam/m/lstm_105/lstm_cell_110/kernel
4:2x2$Adam/v/lstm_105/lstm_cell_110/kernel
>:<x2.Adam/m/lstm_105/lstm_cell_110/recurrent_kernel
>:<x2.Adam/v/lstm_105/lstm_cell_110/recurrent_kernel
.:,x2"Adam/m/lstm_105/lstm_cell_110/bias
.:,x2"Adam/v/lstm_105/lstm_cell_110/bias
4:2x2$Adam/m/lstm_106/lstm_cell_111/kernel
4:2x2$Adam/v/lstm_106/lstm_cell_111/kernel
>:<x2.Adam/m/lstm_106/lstm_cell_111/recurrent_kernel
>:<x2.Adam/v/lstm_106/lstm_cell_111/recurrent_kernel
.:,x2"Adam/m/lstm_106/lstm_cell_111/bias
.:,x2"Adam/v/lstm_106/lstm_cell_111/bias
4:2x2$Adam/m/lstm_107/lstm_cell_112/kernel
4:2x2$Adam/v/lstm_107/lstm_cell_112/kernel
>:<x2.Adam/m/lstm_107/lstm_cell_112/recurrent_kernel
>:<x2.Adam/v/lstm_107/lstm_cell_112/recurrent_kernel
.:,x2"Adam/m/lstm_107/lstm_cell_112/bias
.:,x2"Adam/v/lstm_107/lstm_cell_112/bias
&:$2Adam/m/dense_85/kernel
&:$2Adam/v/dense_85/kernel
 :2Adam/m/dense_85/bias
 :2Adam/v/dense_85/bias
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
0__inference_lstm_cell_110_layer_call_fn_23135928inputsstates_0states_1"�
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
0__inference_lstm_cell_110_layer_call_fn_23135945inputsstates_0states_1"�
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23135977inputsstates_0states_1"�
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23136009inputsstates_0states_1"�
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
0__inference_lstm_cell_111_layer_call_fn_23136026inputsstates_0states_1"�
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
0__inference_lstm_cell_111_layer_call_fn_23136043inputsstates_0states_1"�
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136075inputsstates_0states_1"�
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136107inputsstates_0states_1"�
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
0__inference_lstm_cell_112_layer_call_fn_23136124inputsstates_0states_1"�
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
0__inference_lstm_cell_112_layer_call_fn_23136141inputsstates_0states_1"�
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136173inputsstates_0states_1"�
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136205inputsstates_0states_1"�
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
#__inference__wrapped_model_231307919:;<=>?@A78;�8
1�.
,�)
lstm_105_input���������
� "3�0
.
dense_85"�
dense_85����������
F__inference_dense_85_layer_call_and_return_conditional_losses_23135911c78/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_85_layer_call_fn_23135901X78/�,
%�"
 �
inputs���������
� "!�
unknown����������
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135880c3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
H__inference_dropout_68_layer_call_and_return_conditional_losses_23135892c3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
-__inference_dropout_68_layer_call_fn_23135870X3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
-__inference_dropout_68_layer_call_fn_23135875X3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134196�9:;O�L
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134339�9:;O�L
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134482x9:;?�<
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
F__inference_lstm_105_layer_call_and_return_conditional_losses_23134625x9:;?�<
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
+__inference_lstm_105_layer_call_fn_23134020�9:;O�L
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
+__inference_lstm_105_layer_call_fn_23134031�9:;O�L
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
+__inference_lstm_105_layer_call_fn_23134042m9:;?�<
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
+__inference_lstm_105_layer_call_fn_23134053m9:;?�<
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134812�<=>O�L
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23134955�<=>O�L
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135098x<=>?�<
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
F__inference_lstm_106_layer_call_and_return_conditional_losses_23135241x<=>?�<
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
+__inference_lstm_106_layer_call_fn_23134636�<=>O�L
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
+__inference_lstm_106_layer_call_fn_23134647�<=>O�L
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
+__inference_lstm_106_layer_call_fn_23134658m<=>?�<
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
+__inference_lstm_106_layer_call_fn_23134669m<=>?�<
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135430�?@AO�L
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135575�?@AO�L
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135720t?@A?�<
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
F__inference_lstm_107_layer_call_and_return_conditional_losses_23135865t?@A?�<
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
+__inference_lstm_107_layer_call_fn_23135252y?@AO�L
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
+__inference_lstm_107_layer_call_fn_23135263y?@AO�L
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
+__inference_lstm_107_layer_call_fn_23135274i?@A?�<
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
+__inference_lstm_107_layer_call_fn_23135285i?@A?�<
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23135977�9:;��}
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
K__inference_lstm_cell_110_layer_call_and_return_conditional_losses_23136009�9:;��}
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
0__inference_lstm_cell_110_layer_call_fn_23135928�9:;��}
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
0__inference_lstm_cell_110_layer_call_fn_23135945�9:;��}
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136075�<=>��}
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
K__inference_lstm_cell_111_layer_call_and_return_conditional_losses_23136107�<=>��}
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
0__inference_lstm_cell_111_layer_call_fn_23136026�<=>��}
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
0__inference_lstm_cell_111_layer_call_fn_23136043�<=>��}
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136173�?@A��}
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
K__inference_lstm_cell_112_layer_call_and_return_conditional_losses_23136205�?@A��}
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
0__inference_lstm_cell_112_layer_call_fn_23136124�?@A��}
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
0__inference_lstm_cell_112_layer_call_fn_23136141�?@A��}
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133026�9:;<=>?@A78C�@
9�6
,�)
lstm_105_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133057�9:;<=>?@A78C�@
9�6
,�)
lstm_105_input���������
p

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_87_layer_call_and_return_conditional_losses_23133572x9:;<=>?@A78;�8
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
K__inference_sequential_87_layer_call_and_return_conditional_losses_23134009x9:;<=>?@A78;�8
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
0__inference_sequential_87_layer_call_fn_23132353u9:;<=>?@A78C�@
9�6
,�)
lstm_105_input���������
p 

 
� "!�
unknown����������
0__inference_sequential_87_layer_call_fn_23132995u9:;<=>?@A78C�@
9�6
,�)
lstm_105_input���������
p

 
� "!�
unknown����������
0__inference_sequential_87_layer_call_fn_23133115m9:;<=>?@A78;�8
1�.
$�!
inputs���������
p 

 
� "!�
unknown����������
0__inference_sequential_87_layer_call_fn_23133142m9:;<=>?@A78;�8
1�.
$�!
inputs���������
p

 
� "!�
unknown����������
&__inference_signature_wrapper_23133088�9:;<=>?@A78M�J
� 
C�@
>
lstm_105_input,�)
lstm_105_input���������"3�0
.
dense_85"�
dense_85���������