§≈1
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
И"serve*2.11.02v2.11.0-rc2-15-g6290819256d8¶√.
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
Adam/v/dense_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_89/bias
y
(Adam/v/dense_89/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_89/bias*
_output_shapes
:*
dtype0
А
Adam/m/dense_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_89/bias
y
(Adam/m/dense_89/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_89/bias*
_output_shapes
:*
dtype0
Й
Adam/v/dense_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	і*'
shared_nameAdam/v/dense_89/kernel
В
*Adam/v/dense_89/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_89/kernel*
_output_shapes
:	і*
dtype0
Й
Adam/m/dense_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	і*'
shared_nameAdam/m/dense_89/kernel
В
*Adam/m/dense_89/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_89/kernel*
_output_shapes
:	і*
dtype0
Э
"Adam/v/lstm_119/lstm_cell_124/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*3
shared_name$"Adam/v/lstm_119/lstm_cell_124/bias
Ц
6Adam/v/lstm_119/lstm_cell_124/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_119/lstm_cell_124/bias*
_output_shapes	
:–*
dtype0
Э
"Adam/m/lstm_119/lstm_cell_124/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*3
shared_name$"Adam/m/lstm_119/lstm_cell_124/bias
Ц
6Adam/m/lstm_119/lstm_cell_124/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_119/lstm_cell_124/bias*
_output_shapes	
:–*
dtype0
Ї
.Adam/v/lstm_119/lstm_cell_124/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*?
shared_name0.Adam/v/lstm_119/lstm_cell_124/recurrent_kernel
≥
BAdam/v/lstm_119/lstm_cell_124/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_119/lstm_cell_124/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
Ї
.Adam/m/lstm_119/lstm_cell_124/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*?
shared_name0.Adam/m/lstm_119/lstm_cell_124/recurrent_kernel
≥
BAdam/m/lstm_119/lstm_cell_124/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_119/lstm_cell_124/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
¶
$Adam/v/lstm_119/lstm_cell_124/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*5
shared_name&$Adam/v/lstm_119/lstm_cell_124/kernel
Я
8Adam/v/lstm_119/lstm_cell_124/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_119/lstm_cell_124/kernel* 
_output_shapes
:
і–*
dtype0
¶
$Adam/m/lstm_119/lstm_cell_124/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*5
shared_name&$Adam/m/lstm_119/lstm_cell_124/kernel
Я
8Adam/m/lstm_119/lstm_cell_124/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_119/lstm_cell_124/kernel* 
_output_shapes
:
і–*
dtype0
Э
"Adam/v/lstm_118/lstm_cell_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*3
shared_name$"Adam/v/lstm_118/lstm_cell_123/bias
Ц
6Adam/v/lstm_118/lstm_cell_123/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_118/lstm_cell_123/bias*
_output_shapes	
:–*
dtype0
Э
"Adam/m/lstm_118/lstm_cell_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*3
shared_name$"Adam/m/lstm_118/lstm_cell_123/bias
Ц
6Adam/m/lstm_118/lstm_cell_123/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_118/lstm_cell_123/bias*
_output_shapes	
:–*
dtype0
Ї
.Adam/v/lstm_118/lstm_cell_123/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*?
shared_name0.Adam/v/lstm_118/lstm_cell_123/recurrent_kernel
≥
BAdam/v/lstm_118/lstm_cell_123/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_118/lstm_cell_123/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
Ї
.Adam/m/lstm_118/lstm_cell_123/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*?
shared_name0.Adam/m/lstm_118/lstm_cell_123/recurrent_kernel
≥
BAdam/m/lstm_118/lstm_cell_123/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_118/lstm_cell_123/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
¶
$Adam/v/lstm_118/lstm_cell_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*5
shared_name&$Adam/v/lstm_118/lstm_cell_123/kernel
Я
8Adam/v/lstm_118/lstm_cell_123/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_118/lstm_cell_123/kernel* 
_output_shapes
:
і–*
dtype0
¶
$Adam/m/lstm_118/lstm_cell_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*5
shared_name&$Adam/m/lstm_118/lstm_cell_123/kernel
Я
8Adam/m/lstm_118/lstm_cell_123/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_118/lstm_cell_123/kernel* 
_output_shapes
:
і–*
dtype0
Э
"Adam/v/lstm_117/lstm_cell_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*3
shared_name$"Adam/v/lstm_117/lstm_cell_122/bias
Ц
6Adam/v/lstm_117/lstm_cell_122/bias/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_117/lstm_cell_122/bias*
_output_shapes	
:–*
dtype0
Э
"Adam/m/lstm_117/lstm_cell_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*3
shared_name$"Adam/m/lstm_117/lstm_cell_122/bias
Ц
6Adam/m/lstm_117/lstm_cell_122/bias/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_117/lstm_cell_122/bias*
_output_shapes	
:–*
dtype0
Ї
.Adam/v/lstm_117/lstm_cell_122/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*?
shared_name0.Adam/v/lstm_117/lstm_cell_122/recurrent_kernel
≥
BAdam/v/lstm_117/lstm_cell_122/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/v/lstm_117/lstm_cell_122/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
Ї
.Adam/m/lstm_117/lstm_cell_122/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*?
shared_name0.Adam/m/lstm_117/lstm_cell_122/recurrent_kernel
≥
BAdam/m/lstm_117/lstm_cell_122/recurrent_kernel/Read/ReadVariableOpReadVariableOp.Adam/m/lstm_117/lstm_cell_122/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
•
$Adam/v/lstm_117/lstm_cell_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	–*5
shared_name&$Adam/v/lstm_117/lstm_cell_122/kernel
Ю
8Adam/v/lstm_117/lstm_cell_122/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/lstm_117/lstm_cell_122/kernel*
_output_shapes
:	–*
dtype0
•
$Adam/m/lstm_117/lstm_cell_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	–*5
shared_name&$Adam/m/lstm_117/lstm_cell_122/kernel
Ю
8Adam/m/lstm_117/lstm_cell_122/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/lstm_117/lstm_cell_122/kernel*
_output_shapes
:	–*
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
П
lstm_119/lstm_cell_124/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*,
shared_namelstm_119/lstm_cell_124/bias
И
/lstm_119/lstm_cell_124/bias/Read/ReadVariableOpReadVariableOplstm_119/lstm_cell_124/bias*
_output_shapes	
:–*
dtype0
ђ
'lstm_119/lstm_cell_124/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*8
shared_name)'lstm_119/lstm_cell_124/recurrent_kernel
•
;lstm_119/lstm_cell_124/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_119/lstm_cell_124/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
Ш
lstm_119/lstm_cell_124/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*.
shared_namelstm_119/lstm_cell_124/kernel
С
1lstm_119/lstm_cell_124/kernel/Read/ReadVariableOpReadVariableOplstm_119/lstm_cell_124/kernel* 
_output_shapes
:
і–*
dtype0
П
lstm_118/lstm_cell_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*,
shared_namelstm_118/lstm_cell_123/bias
И
/lstm_118/lstm_cell_123/bias/Read/ReadVariableOpReadVariableOplstm_118/lstm_cell_123/bias*
_output_shapes	
:–*
dtype0
ђ
'lstm_118/lstm_cell_123/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*8
shared_name)'lstm_118/lstm_cell_123/recurrent_kernel
•
;lstm_118/lstm_cell_123/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_118/lstm_cell_123/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
Ш
lstm_118/lstm_cell_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*.
shared_namelstm_118/lstm_cell_123/kernel
С
1lstm_118/lstm_cell_123/kernel/Read/ReadVariableOpReadVariableOplstm_118/lstm_cell_123/kernel* 
_output_shapes
:
і–*
dtype0
П
lstm_117/lstm_cell_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:–*,
shared_namelstm_117/lstm_cell_122/bias
И
/lstm_117/lstm_cell_122/bias/Read/ReadVariableOpReadVariableOplstm_117/lstm_cell_122/bias*
_output_shapes	
:–*
dtype0
ђ
'lstm_117/lstm_cell_122/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
і–*8
shared_name)'lstm_117/lstm_cell_122/recurrent_kernel
•
;lstm_117/lstm_cell_122/recurrent_kernel/Read/ReadVariableOpReadVariableOp'lstm_117/lstm_cell_122/recurrent_kernel* 
_output_shapes
:
і–*
dtype0
Ч
lstm_117/lstm_cell_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	–*.
shared_namelstm_117/lstm_cell_122/kernel
Р
1lstm_117/lstm_cell_122/kernel/Read/ReadVariableOpReadVariableOplstm_117/lstm_cell_122/kernel*
_output_shapes
:	–*
dtype0
r
dense_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_89/bias
k
!dense_89/bias/Read/ReadVariableOpReadVariableOpdense_89/bias*
_output_shapes
:*
dtype0
{
dense_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	і* 
shared_namedense_89/kernel
t
#dense_89/kernel/Read/ReadVariableOpReadVariableOpdense_89/kernel*
_output_shapes
:	і*
dtype0
Й
serving_default_lstm_117_inputPlaceholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
†
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_117_inputlstm_117/lstm_cell_122/kernel'lstm_117/lstm_cell_122/recurrent_kernellstm_117/lstm_cell_122/biaslstm_118/lstm_cell_123/kernel'lstm_118/lstm_cell_123/recurrent_kernellstm_118/lstm_cell_123/biaslstm_119/lstm_cell_124/kernel'lstm_119/lstm_cell_124/recurrent_kernellstm_119/lstm_cell_124/biasdense_89/kerneldense_89/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_23346756

NoOpNoOp
ЭY
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЎX
valueќXBЋX BƒX
х
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
Ѕ
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
Ѕ
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
Ѕ
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
•
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator* 
¶
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
∞
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
Б
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
Я

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
г
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
Я

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
ж
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses
Б_random_generator
В
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
•
Гstates
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
:
Йtrace_0
Кtrace_1
Лtrace_2
Мtrace_3* 
:
Нtrace_0
Оtrace_1
Пtrace_2
Рtrace_3* 
* 
л
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator
Ш
state_size

?kernel
@recurrent_kernel
Abias*
* 
* 
* 
* 
Ц
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

Юtrace_0
Яtrace_1* 

†trace_0
°trace_1* 
* 

70
81*

70
81*
* 
Ш
Ґnon_trainable_variables
£layers
§metrics
 •layer_regularization_losses
¶layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

Іtrace_0* 

®trace_0* 
_Y
VARIABLE_VALUEdense_89/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_89/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_117/lstm_cell_122/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_117/lstm_cell_122/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_117/lstm_cell_122/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_118/lstm_cell_123/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_118/lstm_cell_123/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_118/lstm_cell_123/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUElstm_119/lstm_cell_124/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'lstm_119/lstm_cell_124/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_119/lstm_cell_124/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

©0
™1*
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
»
P0
Ђ1
ђ2
≠3
Ѓ4
ѓ5
∞6
±7
≤8
≥9
і10
µ11
ґ12
Ј13
Є14
є15
Ї16
ї17
Љ18
љ19
Њ20
њ21
ј22*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
]
Ђ0
≠1
ѓ2
±3
≥4
µ5
Ј6
є7
ї8
љ9
њ10*
]
ђ0
Ѓ1
∞2
≤3
і4
ґ5
Є6
Ї7
Љ8
Њ9
ј10*
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
Ш
Ѕnon_trainable_variables
¬layers
√metrics
 ƒlayer_regularization_losses
≈layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

∆trace_0
«trace_1* 

»trace_0
…trace_1* 
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
Ъ
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

ѕtrace_0
–trace_1* 

—trace_0
“trace_1* 
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
Ю
”non_trainable_variables
‘layers
’metrics
 ÷layer_regularization_losses
„layer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses*

Ўtrace_0
ўtrace_1* 

Џtrace_0
џtrace_1* 
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
№	variables
Ё	keras_api

ёtotal

яcount*
M
а	variables
б	keras_api

вtotal

гcount
д
_fn_kwargs*
oi
VARIABLE_VALUE$Adam/m/lstm_117/lstm_cell_122/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/lstm_117/lstm_cell_122/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/m/lstm_117/lstm_cell_122/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/v/lstm_117/lstm_cell_122/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/lstm_117/lstm_cell_122/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_117/lstm_cell_122/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/lstm_118/lstm_cell_123/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/lstm_118/lstm_cell_123/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.Adam/m/lstm_118/lstm_cell_123/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/lstm_118/lstm_cell_123/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/lstm_118/lstm_cell_123/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/lstm_118/lstm_cell_123/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/lstm_119/lstm_cell_124/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/lstm_119/lstm_cell_124/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/m/lstm_119/lstm_cell_124/recurrent_kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/lstm_119/lstm_cell_124/recurrent_kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/lstm_119/lstm_cell_124/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/lstm_119/lstm_cell_124/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_89/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_89/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_89/bias2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_89/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
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
ё0
я1*

№	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

в0
г1*

а	variables*
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
С
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_89/kernel/Read/ReadVariableOp!dense_89/bias/Read/ReadVariableOp1lstm_117/lstm_cell_122/kernel/Read/ReadVariableOp;lstm_117/lstm_cell_122/recurrent_kernel/Read/ReadVariableOp/lstm_117/lstm_cell_122/bias/Read/ReadVariableOp1lstm_118/lstm_cell_123/kernel/Read/ReadVariableOp;lstm_118/lstm_cell_123/recurrent_kernel/Read/ReadVariableOp/lstm_118/lstm_cell_123/bias/Read/ReadVariableOp1lstm_119/lstm_cell_124/kernel/Read/ReadVariableOp;lstm_119/lstm_cell_124/recurrent_kernel/Read/ReadVariableOp/lstm_119/lstm_cell_124/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp8Adam/m/lstm_117/lstm_cell_122/kernel/Read/ReadVariableOp8Adam/v/lstm_117/lstm_cell_122/kernel/Read/ReadVariableOpBAdam/m/lstm_117/lstm_cell_122/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_117/lstm_cell_122/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_117/lstm_cell_122/bias/Read/ReadVariableOp6Adam/v/lstm_117/lstm_cell_122/bias/Read/ReadVariableOp8Adam/m/lstm_118/lstm_cell_123/kernel/Read/ReadVariableOp8Adam/v/lstm_118/lstm_cell_123/kernel/Read/ReadVariableOpBAdam/m/lstm_118/lstm_cell_123/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_118/lstm_cell_123/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_118/lstm_cell_123/bias/Read/ReadVariableOp6Adam/v/lstm_118/lstm_cell_123/bias/Read/ReadVariableOp8Adam/m/lstm_119/lstm_cell_124/kernel/Read/ReadVariableOp8Adam/v/lstm_119/lstm_cell_124/kernel/Read/ReadVariableOpBAdam/m/lstm_119/lstm_cell_124/recurrent_kernel/Read/ReadVariableOpBAdam/v/lstm_119/lstm_cell_124/recurrent_kernel/Read/ReadVariableOp6Adam/m/lstm_119/lstm_cell_124/bias/Read/ReadVariableOp6Adam/v/lstm_119/lstm_cell_124/bias/Read/ReadVariableOp*Adam/m/dense_89/kernel/Read/ReadVariableOp*Adam/v/dense_89/kernel/Read/ReadVariableOp(Adam/m/dense_89/bias/Read/ReadVariableOp(Adam/v/dense_89/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*4
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
GPU 2J 8В **
f%R#
!__inference__traced_save_23350013
А
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_89/kerneldense_89/biaslstm_117/lstm_cell_122/kernel'lstm_117/lstm_cell_122/recurrent_kernellstm_117/lstm_cell_122/biaslstm_118/lstm_cell_123/kernel'lstm_118/lstm_cell_123/recurrent_kernellstm_118/lstm_cell_123/biaslstm_119/lstm_cell_124/kernel'lstm_119/lstm_cell_124/recurrent_kernellstm_119/lstm_cell_124/bias	iterationlearning_rate$Adam/m/lstm_117/lstm_cell_122/kernel$Adam/v/lstm_117/lstm_cell_122/kernel.Adam/m/lstm_117/lstm_cell_122/recurrent_kernel.Adam/v/lstm_117/lstm_cell_122/recurrent_kernel"Adam/m/lstm_117/lstm_cell_122/bias"Adam/v/lstm_117/lstm_cell_122/bias$Adam/m/lstm_118/lstm_cell_123/kernel$Adam/v/lstm_118/lstm_cell_123/kernel.Adam/m/lstm_118/lstm_cell_123/recurrent_kernel.Adam/v/lstm_118/lstm_cell_123/recurrent_kernel"Adam/m/lstm_118/lstm_cell_123/bias"Adam/v/lstm_118/lstm_cell_123/bias$Adam/m/lstm_119/lstm_cell_124/kernel$Adam/v/lstm_119/lstm_cell_124/kernel.Adam/m/lstm_119/lstm_cell_124/recurrent_kernel.Adam/v/lstm_119/lstm_cell_124/recurrent_kernel"Adam/m/lstm_119/lstm_cell_124/bias"Adam/v/lstm_119/lstm_cell_124/biasAdam/m/dense_89/kernelAdam/v/dense_89/kernelAdam/m/dense_89/biasAdam/v/dense_89/biastotal_1count_1totalcount*3
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_23350140°к,
≤
ї
+__inference_lstm_117_layer_call_fn_23347688
inputs_0
unknown:	–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23344609}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і`
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
МL
¶
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349388

inputs@
,lstm_cell_124_matmul_readvariableop_resource:
і–B
.lstm_cell_124_matmul_1_readvariableop_resource:
і–<
-lstm_cell_124_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_124/BiasAdd/ReadVariableOpҐ#lstm_cell_124/MatMul/ReadVariableOpҐ%lstm_cell_124/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_124/MatMul/ReadVariableOpReadVariableOp,lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_124/MatMulMatMulstrided_slice_2:output:0+lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_124/MatMul_1MatMulzeros:output:0-lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_124/addAddV2lstm_cell_124/MatMul:product:0 lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_124/BiasAddBiasAddlstm_cell_124/add:z:0,lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_124/splitSplit&lstm_cell_124/split/split_dim:output:0lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_124/SigmoidSigmoidlstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_1Sigmoidlstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_124/mulMullstm_cell_124/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_124/ReluRelulstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_124/mul_1Mullstm_cell_124/Sigmoid:y:0 lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_124/add_1AddV2lstm_cell_124/mul:z:0lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_2Sigmoidlstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_124/Relu_1Relulstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_124/mul_2Mullstm_cell_124/Sigmoid_2:y:0"lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_124_matmul_readvariableop_resource.lstm_cell_124_matmul_1_readvariableop_resource-lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23349303*
condR
while_cond_23349302*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_124/BiasAdd/ReadVariableOp$^lstm_cell_124/MatMul/ReadVariableOp&^lstm_cell_124/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_124/BiasAdd/ReadVariableOp$lstm_cell_124/BiasAdd/ReadVariableOp2J
#lstm_cell_124/MatMul/ReadVariableOp#lstm_cell_124/MatMul/ReadVariableOp2N
%lstm_cell_124/MatMul_1/ReadVariableOp%lstm_cell_124/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
И:
я
while_body_23349303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_124_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_124_matmul_readvariableop_resource:
і–H
4while_lstm_cell_124_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_124/BiasAdd/ReadVariableOpҐ)while/lstm_cell_124/MatMul/ReadVariableOpҐ+while/lstm_cell_124/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_124/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_124/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_124/addAddV2$while/lstm_cell_124/MatMul:product:0&while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_124/BiasAddBiasAddwhile/lstm_cell_124/add:z:02while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_124/splitSplit,while/lstm_cell_124/split/split_dim:output:0$while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_124/SigmoidSigmoid"while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_1Sigmoid"while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_124/mulMul!while/lstm_cell_124/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_124/ReluRelu"while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_124/mul_1Mulwhile/lstm_cell_124/Sigmoid:y:0&while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_124/add_1AddV2while/lstm_cell_124/mul:z:0while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_2Sigmoid"while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_124/Relu_1Reluwhile/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_124/mul_2Mul!while/lstm_cell_124/Sigmoid_2:y:0(while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_124/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_124/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_124/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_124/BiasAdd/ReadVariableOp*^while/lstm_cell_124/MatMul/ReadVariableOp,^while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_124_biasadd_readvariableop_resource5while_lstm_cell_124_biasadd_readvariableop_resource_0"n
4while_lstm_cell_124_matmul_1_readvariableop_resource6while_lstm_cell_124_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_124_matmul_readvariableop_resource4while_lstm_cell_124_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_124/BiasAdd/ReadVariableOp*while/lstm_cell_124/BiasAdd/ReadVariableOp2V
)while/lstm_cell_124/MatMul/ReadVariableOp)while/lstm_cell_124/MatMul/ReadVariableOp2Z
+while/lstm_cell_124/MatMul_1/ReadVariableOp+while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
€
ы
0__inference_lstm_cell_123_layer_call_fn_23349711

inputs
states_0
states_1
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identity

identity_1

identity_2ИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23345022p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
√
Ќ
while_cond_23344730
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23344730___redundant_placeholder06
2while_while_cond_23344730___redundant_placeholder16
2while_while_cond_23344730___redundant_placeholder26
2while_while_cond_23344730___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
√
Ќ
while_cond_23349447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23349447___redundant_placeholder06
2while_while_cond_23349447___redundant_placeholder16
2while_while_cond_23349447___redundant_placeholder26
2while_while_cond_23349447___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
Ђ#
ъ
while_body_23344731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_122_23344755_0:	–2
while_lstm_cell_122_23344757_0:
і–-
while_lstm_cell_122_23344759_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_122_23344755:	–0
while_lstm_cell_122_23344757:
і–+
while_lstm_cell_122_23344759:	–ИҐ+while/lstm_cell_122/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0√
+while/lstm_cell_122/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_122_23344755_0while_lstm_cell_122_23344757_0while_lstm_cell_122_23344759_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344672Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_122/StatefulPartitionedCall:output:0*
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
: Т
while/Identity_4Identity4while/lstm_cell_122/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іТ
while/Identity_5Identity4while/lstm_cell_122/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іz

while/NoOpNoOp,^while/lstm_cell_122/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_122_23344755while_lstm_cell_122_23344755_0">
while_lstm_cell_122_23344757while_lstm_cell_122_23344757_0">
while_lstm_cell_122_23344759while_lstm_cell_122_23344759_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Z
+while/lstm_cell_122/StatefulPartitionedCall+while/lstm_cell_122/StatefulPartitionedCall: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
 $
ь
while_body_23345434
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_124_23345458_0:
і–2
while_lstm_cell_124_23345460_0:
і–-
while_lstm_cell_124_23345462_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_124_23345458:
і–0
while_lstm_cell_124_23345460:
і–+
while_lstm_cell_124_23345462:	–ИҐ+while/lstm_cell_124/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0√
+while/lstm_cell_124/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_124_23345458_0while_lstm_cell_124_23345460_0while_lstm_cell_124_23345462_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345374r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Е
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_124/StatefulPartitionedCall:output:0*
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
: Т
while/Identity_4Identity4while/lstm_cell_124/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іТ
while/Identity_5Identity4while/lstm_cell_124/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іz

while/NoOpNoOp,^while/lstm_cell_124/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_124_23345458while_lstm_cell_124_23345458_0">
while_lstm_cell_124_23345460while_lstm_cell_124_23345460_0">
while_lstm_cell_124_23345462while_lstm_cell_124_23345462_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Z
+while/lstm_cell_124/StatefulPartitionedCall+while/lstm_cell_124/StatefulPartitionedCall: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
И
є
+__inference_lstm_117_layer_call_fn_23347721

inputs
unknown:	–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23346542t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і`
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
ƒK
®
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348480
inputs_0@
,lstm_cell_123_matmul_readvariableop_resource:
і–B
.lstm_cell_123_matmul_1_readvariableop_resource:
і–<
-lstm_cell_123_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_123/BiasAdd/ReadVariableOpҐ#lstm_cell_123/MatMul/ReadVariableOpҐ%lstm_cell_123/MatMul_1/ReadVariableOpҐwhile=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_123/MatMul/ReadVariableOpReadVariableOp,lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_123/MatMulMatMulstrided_slice_2:output:0+lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_123/MatMul_1MatMulzeros:output:0-lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_123/addAddV2lstm_cell_123/MatMul:product:0 lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_123/BiasAddBiasAddlstm_cell_123/add:z:0,lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_123/splitSplit&lstm_cell_123/split/split_dim:output:0lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_123/SigmoidSigmoidlstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_1Sigmoidlstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_123/mulMullstm_cell_123/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_123/ReluRelulstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_123/mul_1Mullstm_cell_123/Sigmoid:y:0 lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_123/add_1AddV2lstm_cell_123/mul:z:0lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_2Sigmoidlstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_123/Relu_1Relulstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_123/mul_2Mullstm_cell_123/Sigmoid_2:y:0"lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_123_matmul_readvariableop_resource.lstm_cell_123_matmul_1_readvariableop_resource-lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23348396*
condR
while_cond_23348395*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і√
NoOpNoOp%^lstm_cell_123/BiasAdd/ReadVariableOp$^lstm_cell_123/MatMul/ReadVariableOp&^lstm_cell_123/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2L
$lstm_cell_123/BiasAdd/ReadVariableOp$lstm_cell_123/BiasAdd/ReadVariableOp2J
#lstm_cell_123/MatMul/ReadVariableOp#lstm_cell_123/MatMul/ReadVariableOp2N
%lstm_cell_123/MatMul_1/ReadVariableOp%lstm_cell_123/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
Ђ#
ъ
while_body_23344540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_122_23344564_0:	–2
while_lstm_cell_122_23344566_0:
і–-
while_lstm_cell_122_23344568_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_122_23344564:	–0
while_lstm_cell_122_23344566:
і–+
while_lstm_cell_122_23344568:	–ИҐ+while/lstm_cell_122/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0√
+while/lstm_cell_122/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_122_23344564_0while_lstm_cell_122_23344566_0while_lstm_cell_122_23344568_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344526Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_122/StatefulPartitionedCall:output:0*
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
: Т
while/Identity_4Identity4while/lstm_cell_122/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іТ
while/Identity_5Identity4while/lstm_cell_122/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іz

while/NoOpNoOp,^while/lstm_cell_122/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_122_23344564while_lstm_cell_122_23344564_0">
while_lstm_cell_122_23344566while_lstm_cell_122_23344566_0">
while_lstm_cell_122_23344568while_lstm_cell_122_23344568_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Z
+while/lstm_cell_122/StatefulPartitionedCall+while/lstm_cell_122/StatefulPartitionedCall: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
И:
я
while_body_23349013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_124_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_124_matmul_readvariableop_resource:
і–H
4while_lstm_cell_124_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_124/BiasAdd/ReadVariableOpҐ)while/lstm_cell_124/MatMul/ReadVariableOpҐ+while/lstm_cell_124/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_124/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_124/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_124/addAddV2$while/lstm_cell_124/MatMul:product:0&while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_124/BiasAddBiasAddwhile/lstm_cell_124/add:z:02while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_124/splitSplit,while/lstm_cell_124/split/split_dim:output:0$while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_124/SigmoidSigmoid"while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_1Sigmoid"while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_124/mulMul!while/lstm_cell_124/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_124/ReluRelu"while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_124/mul_1Mulwhile/lstm_cell_124/Sigmoid:y:0&while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_124/add_1AddV2while/lstm_cell_124/mul:z:0while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_2Sigmoid"while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_124/Relu_1Reluwhile/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_124/mul_2Mul!while/lstm_cell_124/Sigmoid_2:y:0(while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_124/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_124/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_124/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_124/BiasAdd/ReadVariableOp*^while/lstm_cell_124/MatMul/ReadVariableOp,^while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_124_biasadd_readvariableop_resource5while_lstm_cell_124_biasadd_readvariableop_resource_0"n
4while_lstm_cell_124_matmul_1_readvariableop_resource6while_lstm_cell_124_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_124_matmul_readvariableop_resource4while_lstm_cell_124_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_124/BiasAdd/ReadVariableOp*while/lstm_cell_124/BiasAdd/ReadVariableOp2V
)while/lstm_cell_124/MatMul/ReadVariableOp)while/lstm_cell_124/MatMul/ReadVariableOp2Z
+while/lstm_cell_124/MatMul_1/ReadVariableOp+while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
€
ы
0__inference_lstm_cell_123_layer_call_fn_23349694

inputs
states_0
states_1
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identity

identity_1

identity_2ИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23344876p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
Л
Ї
+__inference_lstm_118_layer_call_fn_23348326

inputs
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23345812t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
√
Ќ
while_cond_23349157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23349157___redundant_placeholder06
2while_while_cond_23349157___redundant_placeholder16
2while_while_cond_23349157___redundant_placeholder26
2while_while_cond_23349157___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
Х

g
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349560

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€і*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€іT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€і"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€і:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
юS
¬
*sequential_91_lstm_118_while_body_23344227J
Fsequential_91_lstm_118_while_sequential_91_lstm_118_while_loop_counterP
Lsequential_91_lstm_118_while_sequential_91_lstm_118_while_maximum_iterations,
(sequential_91_lstm_118_while_placeholder.
*sequential_91_lstm_118_while_placeholder_1.
*sequential_91_lstm_118_while_placeholder_2.
*sequential_91_lstm_118_while_placeholder_3I
Esequential_91_lstm_118_while_sequential_91_lstm_118_strided_slice_1_0Ж
Бsequential_91_lstm_118_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_118_tensorarrayunstack_tensorlistfromtensor_0_
Ksequential_91_lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0:
і–a
Msequential_91_lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–[
Lsequential_91_lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0:	–)
%sequential_91_lstm_118_while_identity+
'sequential_91_lstm_118_while_identity_1+
'sequential_91_lstm_118_while_identity_2+
'sequential_91_lstm_118_while_identity_3+
'sequential_91_lstm_118_while_identity_4+
'sequential_91_lstm_118_while_identity_5G
Csequential_91_lstm_118_while_sequential_91_lstm_118_strided_slice_1Г
sequential_91_lstm_118_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_118_tensorarrayunstack_tensorlistfromtensor]
Isequential_91_lstm_118_while_lstm_cell_123_matmul_readvariableop_resource:
і–_
Ksequential_91_lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource:
і–Y
Jsequential_91_lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐAsequential_91/lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpҐ@sequential_91/lstm_118/while/lstm_cell_123/MatMul/ReadVariableOpҐBsequential_91/lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOpЯ
Nsequential_91/lstm_118/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Ы
@sequential_91/lstm_118/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemБsequential_91_lstm_118_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_118_tensorarrayunstack_tensorlistfromtensor_0(sequential_91_lstm_118_while_placeholderWsequential_91/lstm_118/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0ќ
@sequential_91/lstm_118/while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOpKsequential_91_lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Б
1sequential_91/lstm_118/while/lstm_cell_123/MatMulMatMulGsequential_91/lstm_118/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_91/lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–“
Bsequential_91/lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOpMsequential_91_lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0и
3sequential_91/lstm_118/while/lstm_cell_123/MatMul_1MatMul*sequential_91_lstm_118_while_placeholder_2Jsequential_91/lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ж
.sequential_91/lstm_118/while/lstm_cell_123/addAddV2;sequential_91/lstm_118/while/lstm_cell_123/MatMul:product:0=sequential_91/lstm_118/while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Ћ
Asequential_91/lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOpLsequential_91_lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0п
2sequential_91/lstm_118/while/lstm_cell_123/BiasAddBiasAdd2sequential_91/lstm_118/while/lstm_cell_123/add:z:0Isequential_91/lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–|
:sequential_91/lstm_118/while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
0sequential_91/lstm_118/while/lstm_cell_123/splitSplitCsequential_91/lstm_118/while/lstm_cell_123/split/split_dim:output:0;sequential_91/lstm_118/while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitЂ
2sequential_91/lstm_118/while/lstm_cell_123/SigmoidSigmoid9sequential_91/lstm_118/while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і≠
4sequential_91/lstm_118/while/lstm_cell_123/Sigmoid_1Sigmoid9sequential_91/lstm_118/while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іќ
.sequential_91/lstm_118/while/lstm_cell_123/mulMul8sequential_91/lstm_118/while/lstm_cell_123/Sigmoid_1:y:0*sequential_91_lstm_118_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€і•
/sequential_91/lstm_118/while/lstm_cell_123/ReluRelu9sequential_91/lstm_118/while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іб
0sequential_91/lstm_118/while/lstm_cell_123/mul_1Mul6sequential_91/lstm_118/while/lstm_cell_123/Sigmoid:y:0=sequential_91/lstm_118/while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і÷
0sequential_91/lstm_118/while/lstm_cell_123/add_1AddV22sequential_91/lstm_118/while/lstm_cell_123/mul:z:04sequential_91/lstm_118/while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і≠
4sequential_91/lstm_118/while/lstm_cell_123/Sigmoid_2Sigmoid9sequential_91/lstm_118/while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іҐ
1sequential_91/lstm_118/while/lstm_cell_123/Relu_1Relu4sequential_91/lstm_118/while/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іе
0sequential_91/lstm_118/while/lstm_cell_123/mul_2Mul8sequential_91/lstm_118/while/lstm_cell_123/Sigmoid_2:y:0?sequential_91/lstm_118/while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іҐ
Asequential_91/lstm_118/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_91_lstm_118_while_placeholder_1(sequential_91_lstm_118_while_placeholder4sequential_91/lstm_118/while/lstm_cell_123/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“d
"sequential_91/lstm_118/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :°
 sequential_91/lstm_118/while/addAddV2(sequential_91_lstm_118_while_placeholder+sequential_91/lstm_118/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_91/lstm_118/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :√
"sequential_91/lstm_118/while/add_1AddV2Fsequential_91_lstm_118_while_sequential_91_lstm_118_while_loop_counter-sequential_91/lstm_118/while/add_1/y:output:0*
T0*
_output_shapes
: Ю
%sequential_91/lstm_118/while/IdentityIdentity&sequential_91/lstm_118/while/add_1:z:0"^sequential_91/lstm_118/while/NoOp*
T0*
_output_shapes
: ∆
'sequential_91/lstm_118/while/Identity_1IdentityLsequential_91_lstm_118_while_sequential_91_lstm_118_while_maximum_iterations"^sequential_91/lstm_118/while/NoOp*
T0*
_output_shapes
: Ю
'sequential_91/lstm_118/while/Identity_2Identity$sequential_91/lstm_118/while/add:z:0"^sequential_91/lstm_118/while/NoOp*
T0*
_output_shapes
: Ћ
'sequential_91/lstm_118/while/Identity_3IdentityQsequential_91/lstm_118/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_91/lstm_118/while/NoOp*
T0*
_output_shapes
: ј
'sequential_91/lstm_118/while/Identity_4Identity4sequential_91/lstm_118/while/lstm_cell_123/mul_2:z:0"^sequential_91/lstm_118/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іј
'sequential_91/lstm_118/while/Identity_5Identity4sequential_91/lstm_118/while/lstm_cell_123/add_1:z:0"^sequential_91/lstm_118/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іѓ
!sequential_91/lstm_118/while/NoOpNoOpB^sequential_91/lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpA^sequential_91/lstm_118/while/lstm_cell_123/MatMul/ReadVariableOpC^sequential_91/lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_91_lstm_118_while_identity.sequential_91/lstm_118/while/Identity:output:0"[
'sequential_91_lstm_118_while_identity_10sequential_91/lstm_118/while/Identity_1:output:0"[
'sequential_91_lstm_118_while_identity_20sequential_91/lstm_118/while/Identity_2:output:0"[
'sequential_91_lstm_118_while_identity_30sequential_91/lstm_118/while/Identity_3:output:0"[
'sequential_91_lstm_118_while_identity_40sequential_91/lstm_118/while/Identity_4:output:0"[
'sequential_91_lstm_118_while_identity_50sequential_91/lstm_118/while/Identity_5:output:0"Ъ
Jsequential_91_lstm_118_while_lstm_cell_123_biasadd_readvariableop_resourceLsequential_91_lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0"Ь
Ksequential_91_lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resourceMsequential_91_lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0"Ш
Isequential_91_lstm_118_while_lstm_cell_123_matmul_readvariableop_resourceKsequential_91_lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0"М
Csequential_91_lstm_118_while_sequential_91_lstm_118_strided_slice_1Esequential_91_lstm_118_while_sequential_91_lstm_118_strided_slice_1_0"Е
sequential_91_lstm_118_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_118_tensorarrayunstack_tensorlistfromtensorБsequential_91_lstm_118_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_118_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Ж
Asequential_91/lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpAsequential_91/lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp2Д
@sequential_91/lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp@sequential_91/lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp2И
Bsequential_91/lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOpBsequential_91/lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
Ѓ#
ь
while_body_23345081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_123_23345105_0:
і–2
while_lstm_cell_123_23345107_0:
і–-
while_lstm_cell_123_23345109_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_123_23345105:
і–0
while_lstm_cell_123_23345107:
і–+
while_lstm_cell_123_23345109:	–ИҐ+while/lstm_cell_123/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0√
+while/lstm_cell_123/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_123_23345105_0while_lstm_cell_123_23345107_0while_lstm_cell_123_23345109_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23345022Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_123/StatefulPartitionedCall:output:0*
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
: Т
while/Identity_4Identity4while/lstm_cell_123/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іТ
while/Identity_5Identity4while/lstm_cell_123/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іz

while/NoOpNoOp,^while/lstm_cell_123/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_123_23345105while_lstm_cell_123_23345105_0">
while_lstm_cell_123_23345107while_lstm_cell_123_23345107_0">
while_lstm_cell_123_23345109while_lstm_cell_123_23345109_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Z
+while/lstm_cell_123/StatefulPartitionedCall+while/lstm_cell_123/StatefulPartitionedCall: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
Ы
Љ
+__inference_lstm_119_layer_call_fn_23348931
inputs_0
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345504p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
ЖK
¶
F__inference_lstm_118_layer_call_and_return_conditional_losses_23346377

inputs@
,lstm_cell_123_matmul_readvariableop_resource:
і–B
.lstm_cell_123_matmul_1_readvariableop_resource:
і–<
-lstm_cell_123_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_123/BiasAdd/ReadVariableOpҐ#lstm_cell_123/MatMul/ReadVariableOpҐ%lstm_cell_123/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_123/MatMul/ReadVariableOpReadVariableOp,lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_123/MatMulMatMulstrided_slice_2:output:0+lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_123/MatMul_1MatMulzeros:output:0-lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_123/addAddV2lstm_cell_123/MatMul:product:0 lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_123/BiasAddBiasAddlstm_cell_123/add:z:0,lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_123/splitSplit&lstm_cell_123/split/split_dim:output:0lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_123/SigmoidSigmoidlstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_1Sigmoidlstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_123/mulMullstm_cell_123/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_123/ReluRelulstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_123/mul_1Mullstm_cell_123/Sigmoid:y:0 lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_123/add_1AddV2lstm_cell_123/mul:z:0lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_2Sigmoidlstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_123/Relu_1Relulstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_123/mul_2Mullstm_cell_123/Sigmoid_2:y:0"lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_123_matmul_readvariableop_resource.lstm_cell_123_matmul_1_readvariableop_resource-lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23346293*
condR
while_cond_23346292*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_123/BiasAdd/ReadVariableOp$^lstm_cell_123/MatMul/ReadVariableOp&^lstm_cell_123/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_123/BiasAdd/ReadVariableOp$lstm_cell_123/BiasAdd/ReadVariableOp2J
#lstm_cell_123/MatMul/ReadVariableOp#lstm_cell_123/MatMul/ReadVariableOp2N
%lstm_cell_123/MatMul_1/ReadVariableOp%lstm_cell_123/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
ъ
Л
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349775

inputs
states_0
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
…
Щ
+__inference_dense_89_layer_call_fn_23349569

inputs
unknown:	і
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
F__inference_dense_89_layer_call_and_return_conditional_losses_23345989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€і: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
т
Й
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345226

inputs

states
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates
И:
я
while_body_23349158
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_124_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_124_matmul_readvariableop_resource:
і–H
4while_lstm_cell_124_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_124/BiasAdd/ReadVariableOpҐ)while/lstm_cell_124/MatMul/ReadVariableOpҐ+while/lstm_cell_124/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_124/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_124/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_124/addAddV2$while/lstm_cell_124/MatMul:product:0&while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_124/BiasAddBiasAddwhile/lstm_cell_124/add:z:02while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_124/splitSplit,while/lstm_cell_124/split/split_dim:output:0$while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_124/SigmoidSigmoid"while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_1Sigmoid"while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_124/mulMul!while/lstm_cell_124/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_124/ReluRelu"while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_124/mul_1Mulwhile/lstm_cell_124/Sigmoid:y:0&while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_124/add_1AddV2while/lstm_cell_124/mul:z:0while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_2Sigmoid"while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_124/Relu_1Reluwhile/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_124/mul_2Mul!while/lstm_cell_124/Sigmoid_2:y:0(while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_124/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_124/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_124/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_124/BiasAdd/ReadVariableOp*^while/lstm_cell_124/MatMul/ReadVariableOp,^while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_124_biasadd_readvariableop_resource5while_lstm_cell_124_biasadd_readvariableop_resource_0"n
4while_lstm_cell_124_matmul_1_readvariableop_resource6while_lstm_cell_124_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_124_matmul_readvariableop_resource4while_lstm_cell_124_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_124/BiasAdd/ReadVariableOp*while/lstm_cell_124/BiasAdd/ReadVariableOp2V
)while/lstm_cell_124/MatMul/ReadVariableOp)while/lstm_cell_124/MatMul/ReadVariableOp2Z
+while/lstm_cell_124/MatMul_1/ReadVariableOp+while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
√
Ќ
while_cond_23346126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23346126___redundant_placeholder06
2while_while_cond_23346126___redundant_placeholder16
2while_while_cond_23346126___redundant_placeholder26
2while_while_cond_23346126___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
АK
•
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348293

inputs?
,lstm_cell_122_matmul_readvariableop_resource:	–B
.lstm_cell_122_matmul_1_readvariableop_resource:
і–<
-lstm_cell_122_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_122/BiasAdd/ReadVariableOpҐ#lstm_cell_122/MatMul/ReadVariableOpҐ%lstm_cell_122/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskС
#lstm_cell_122/MatMul/ReadVariableOpReadVariableOp,lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0Ш
lstm_cell_122/MatMulMatMulstrided_slice_2:output:0+lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_122/MatMul_1MatMulzeros:output:0-lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_122/addAddV2lstm_cell_122/MatMul:product:0 lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_122/BiasAddBiasAddlstm_cell_122/add:z:0,lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_122/splitSplit&lstm_cell_122/split/split_dim:output:0lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_122/SigmoidSigmoidlstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_1Sigmoidlstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_122/mulMullstm_cell_122/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_122/ReluRelulstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_122/mul_1Mullstm_cell_122/Sigmoid:y:0 lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_122/add_1AddV2lstm_cell_122/mul:z:0lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_2Sigmoidlstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_122/Relu_1Relulstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_122/mul_2Mullstm_cell_122/Sigmoid_2:y:0"lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_122_matmul_readvariableop_resource.lstm_cell_122_matmul_1_readvariableop_resource-lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23348209*
condR
while_cond_23348208*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_122/BiasAdd/ReadVariableOp$^lstm_cell_122/MatMul/ReadVariableOp&^lstm_cell_122/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2L
$lstm_cell_122/BiasAdd/ReadVariableOp$lstm_cell_122/BiasAdd/ReadVariableOp2J
#lstm_cell_122/MatMul/ReadVariableOp#lstm_cell_122/MatMul/ReadVariableOp2N
%lstm_cell_122/MatMul_1/ReadVariableOp%lstm_cell_122/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
µ
Љ
+__inference_lstm_118_layer_call_fn_23348315
inputs_0
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23345150}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
Ќ	
ш
F__inference_dense_89_layer_call_and_return_conditional_losses_23345989

inputs1
matmul_readvariableop_resource:	і-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	і*
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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€і: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
ы
f
-__inference_dropout_72_layer_call_fn_23349543

inputs
identityИҐStatefulPartitionedCallƒ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_72_layer_call_and_return_conditional_losses_23346051p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€і22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
µ
Љ
+__inference_lstm_118_layer_call_fn_23348304
inputs_0
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23344959}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
«Ї
ё
#__inference__wrapped_model_23344459
lstm_117_inputV
Csequential_91_lstm_117_lstm_cell_122_matmul_readvariableop_resource:	–Y
Esequential_91_lstm_117_lstm_cell_122_matmul_1_readvariableop_resource:
і–S
Dsequential_91_lstm_117_lstm_cell_122_biasadd_readvariableop_resource:	–W
Csequential_91_lstm_118_lstm_cell_123_matmul_readvariableop_resource:
і–Y
Esequential_91_lstm_118_lstm_cell_123_matmul_1_readvariableop_resource:
і–S
Dsequential_91_lstm_118_lstm_cell_123_biasadd_readvariableop_resource:	–W
Csequential_91_lstm_119_lstm_cell_124_matmul_readvariableop_resource:
і–Y
Esequential_91_lstm_119_lstm_cell_124_matmul_1_readvariableop_resource:
і–S
Dsequential_91_lstm_119_lstm_cell_124_biasadd_readvariableop_resource:	–H
5sequential_91_dense_89_matmul_readvariableop_resource:	іD
6sequential_91_dense_89_biasadd_readvariableop_resource:
identityИҐ-sequential_91/dense_89/BiasAdd/ReadVariableOpҐ,sequential_91/dense_89/MatMul/ReadVariableOpҐ;sequential_91/lstm_117/lstm_cell_122/BiasAdd/ReadVariableOpҐ:sequential_91/lstm_117/lstm_cell_122/MatMul/ReadVariableOpҐ<sequential_91/lstm_117/lstm_cell_122/MatMul_1/ReadVariableOpҐsequential_91/lstm_117/whileҐ;sequential_91/lstm_118/lstm_cell_123/BiasAdd/ReadVariableOpҐ:sequential_91/lstm_118/lstm_cell_123/MatMul/ReadVariableOpҐ<sequential_91/lstm_118/lstm_cell_123/MatMul_1/ReadVariableOpҐsequential_91/lstm_118/whileҐ;sequential_91/lstm_119/lstm_cell_124/BiasAdd/ReadVariableOpҐ:sequential_91/lstm_119/lstm_cell_124/MatMul/ReadVariableOpҐ<sequential_91/lstm_119/lstm_cell_124/MatMul_1/ReadVariableOpҐsequential_91/lstm_119/whileZ
sequential_91/lstm_117/ShapeShapelstm_117_input*
T0*
_output_shapes
:t
*sequential_91/lstm_117/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_91/lstm_117/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_91/lstm_117/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ƒ
$sequential_91/lstm_117/strided_sliceStridedSlice%sequential_91/lstm_117/Shape:output:03sequential_91/lstm_117/strided_slice/stack:output:05sequential_91/lstm_117/strided_slice/stack_1:output:05sequential_91/lstm_117/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential_91/lstm_117/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іЄ
#sequential_91/lstm_117/zeros/packedPack-sequential_91/lstm_117/strided_slice:output:0.sequential_91/lstm_117/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_91/lstm_117/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ≤
sequential_91/lstm_117/zerosFill,sequential_91/lstm_117/zeros/packed:output:0+sequential_91/lstm_117/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іj
'sequential_91/lstm_117/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іЉ
%sequential_91/lstm_117/zeros_1/packedPack-sequential_91/lstm_117/strided_slice:output:00sequential_91/lstm_117/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_91/lstm_117/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
sequential_91/lstm_117/zeros_1Fill.sequential_91/lstm_117/zeros_1/packed:output:0-sequential_91/lstm_117/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іz
%sequential_91/lstm_117/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
 sequential_91/lstm_117/transpose	Transposelstm_117_input.sequential_91/lstm_117/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€r
sequential_91/lstm_117/Shape_1Shape$sequential_91/lstm_117/transpose:y:0*
T0*
_output_shapes
:v
,sequential_91/lstm_117/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_117/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_91/lstm_117/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
&sequential_91/lstm_117/strided_slice_1StridedSlice'sequential_91/lstm_117/Shape_1:output:05sequential_91/lstm_117/strided_slice_1/stack:output:07sequential_91/lstm_117/strided_slice_1/stack_1:output:07sequential_91/lstm_117/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_91/lstm_117/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€щ
$sequential_91/lstm_117/TensorArrayV2TensorListReserve;sequential_91/lstm_117/TensorArrayV2/element_shape:output:0/sequential_91/lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Э
Lsequential_91/lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   •
>sequential_91/lstm_117/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_91/lstm_117/transpose:y:0Usequential_91/lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“v
,sequential_91/lstm_117/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_117/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_91/lstm_117/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
&sequential_91/lstm_117/strided_slice_2StridedSlice$sequential_91/lstm_117/transpose:y:05sequential_91/lstm_117/strided_slice_2/stack:output:07sequential_91/lstm_117/strided_slice_2/stack_1:output:07sequential_91/lstm_117/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_maskњ
:sequential_91/lstm_117/lstm_cell_122/MatMul/ReadVariableOpReadVariableOpCsequential_91_lstm_117_lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0Ё
+sequential_91/lstm_117/lstm_cell_122/MatMulMatMul/sequential_91/lstm_117/strided_slice_2:output:0Bsequential_91/lstm_117/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ƒ
<sequential_91/lstm_117/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOpEsequential_91_lstm_117_lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0„
-sequential_91/lstm_117/lstm_cell_122/MatMul_1MatMul%sequential_91/lstm_117/zeros:output:0Dsequential_91/lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–‘
(sequential_91/lstm_117/lstm_cell_122/addAddV25sequential_91/lstm_117/lstm_cell_122/MatMul:product:07sequential_91/lstm_117/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–љ
;sequential_91/lstm_117/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOpDsequential_91_lstm_117_lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ё
,sequential_91/lstm_117/lstm_cell_122/BiasAddBiasAdd,sequential_91/lstm_117/lstm_cell_122/add:z:0Csequential_91/lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–v
4sequential_91/lstm_117/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :©
*sequential_91/lstm_117/lstm_cell_122/splitSplit=sequential_91/lstm_117/lstm_cell_122/split/split_dim:output:05sequential_91/lstm_117/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitЯ
,sequential_91/lstm_117/lstm_cell_122/SigmoidSigmoid3sequential_91/lstm_117/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і°
.sequential_91/lstm_117/lstm_cell_122/Sigmoid_1Sigmoid3sequential_91/lstm_117/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іњ
(sequential_91/lstm_117/lstm_cell_122/mulMul2sequential_91/lstm_117/lstm_cell_122/Sigmoid_1:y:0'sequential_91/lstm_117/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЩ
)sequential_91/lstm_117/lstm_cell_122/ReluRelu3sequential_91/lstm_117/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іѕ
*sequential_91/lstm_117/lstm_cell_122/mul_1Mul0sequential_91/lstm_117/lstm_cell_122/Sigmoid:y:07sequential_91/lstm_117/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іƒ
*sequential_91/lstm_117/lstm_cell_122/add_1AddV2,sequential_91/lstm_117/lstm_cell_122/mul:z:0.sequential_91/lstm_117/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і°
.sequential_91/lstm_117/lstm_cell_122/Sigmoid_2Sigmoid3sequential_91/lstm_117/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЦ
+sequential_91/lstm_117/lstm_cell_122/Relu_1Relu.sequential_91/lstm_117/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і”
*sequential_91/lstm_117/lstm_cell_122/mul_2Mul2sequential_91/lstm_117/lstm_cell_122/Sigmoid_2:y:09sequential_91/lstm_117/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
4sequential_91/lstm_117/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   э
&sequential_91/lstm_117/TensorArrayV2_1TensorListReserve=sequential_91/lstm_117/TensorArrayV2_1/element_shape:output:0/sequential_91/lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“]
sequential_91/lstm_117/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_91/lstm_117/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€k
)sequential_91/lstm_117/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
sequential_91/lstm_117/whileWhile2sequential_91/lstm_117/while/loop_counter:output:08sequential_91/lstm_117/while/maximum_iterations:output:0$sequential_91/lstm_117/time:output:0/sequential_91/lstm_117/TensorArrayV2_1:handle:0%sequential_91/lstm_117/zeros:output:0'sequential_91/lstm_117/zeros_1:output:0/sequential_91/lstm_117/strided_slice_1:output:0Nsequential_91/lstm_117/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_91_lstm_117_lstm_cell_122_matmul_readvariableop_resourceEsequential_91_lstm_117_lstm_cell_122_matmul_1_readvariableop_resourceDsequential_91_lstm_117_lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_91_lstm_117_while_body_23344088*6
cond.R,
*sequential_91_lstm_117_while_cond_23344087*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Ш
Gsequential_91/lstm_117/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   И
9sequential_91/lstm_117/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_91/lstm_117/while:output:3Psequential_91/lstm_117/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0
,sequential_91/lstm_117/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€x
.sequential_91/lstm_117/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_117/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
&sequential_91/lstm_117/strided_slice_3StridedSliceBsequential_91/lstm_117/TensorArrayV2Stack/TensorListStack:tensor:05sequential_91/lstm_117/strided_slice_3/stack:output:07sequential_91/lstm_117/strided_slice_3/stack_1:output:07sequential_91/lstm_117/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_mask|
'sequential_91/lstm_117/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          №
"sequential_91/lstm_117/transpose_1	TransposeBsequential_91/lstm_117/TensorArrayV2Stack/TensorListStack:tensor:00sequential_91/lstm_117/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іr
sequential_91/lstm_117/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    r
sequential_91/lstm_118/ShapeShape&sequential_91/lstm_117/transpose_1:y:0*
T0*
_output_shapes
:t
*sequential_91/lstm_118/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_91/lstm_118/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_91/lstm_118/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ƒ
$sequential_91/lstm_118/strided_sliceStridedSlice%sequential_91/lstm_118/Shape:output:03sequential_91/lstm_118/strided_slice/stack:output:05sequential_91/lstm_118/strided_slice/stack_1:output:05sequential_91/lstm_118/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential_91/lstm_118/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іЄ
#sequential_91/lstm_118/zeros/packedPack-sequential_91/lstm_118/strided_slice:output:0.sequential_91/lstm_118/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_91/lstm_118/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ≤
sequential_91/lstm_118/zerosFill,sequential_91/lstm_118/zeros/packed:output:0+sequential_91/lstm_118/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іj
'sequential_91/lstm_118/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іЉ
%sequential_91/lstm_118/zeros_1/packedPack-sequential_91/lstm_118/strided_slice:output:00sequential_91/lstm_118/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_91/lstm_118/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
sequential_91/lstm_118/zeros_1Fill.sequential_91/lstm_118/zeros_1/packed:output:0-sequential_91/lstm_118/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іz
%sequential_91/lstm_118/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Љ
 sequential_91/lstm_118/transpose	Transpose&sequential_91/lstm_117/transpose_1:y:0.sequential_91/lstm_118/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іr
sequential_91/lstm_118/Shape_1Shape$sequential_91/lstm_118/transpose:y:0*
T0*
_output_shapes
:v
,sequential_91/lstm_118/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_118/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_91/lstm_118/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
&sequential_91/lstm_118/strided_slice_1StridedSlice'sequential_91/lstm_118/Shape_1:output:05sequential_91/lstm_118/strided_slice_1/stack:output:07sequential_91/lstm_118/strided_slice_1/stack_1:output:07sequential_91/lstm_118/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_91/lstm_118/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€щ
$sequential_91/lstm_118/TensorArrayV2TensorListReserve;sequential_91/lstm_118/TensorArrayV2/element_shape:output:0/sequential_91/lstm_118/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Э
Lsequential_91/lstm_118/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   •
>sequential_91/lstm_118/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_91/lstm_118/transpose:y:0Usequential_91/lstm_118/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“v
,sequential_91/lstm_118/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_118/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_91/lstm_118/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
&sequential_91/lstm_118/strided_slice_2StridedSlice$sequential_91/lstm_118/transpose:y:05sequential_91/lstm_118/strided_slice_2/stack:output:07sequential_91/lstm_118/strided_slice_2/stack_1:output:07sequential_91/lstm_118/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskј
:sequential_91/lstm_118/lstm_cell_123/MatMul/ReadVariableOpReadVariableOpCsequential_91_lstm_118_lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ё
+sequential_91/lstm_118/lstm_cell_123/MatMulMatMul/sequential_91/lstm_118/strided_slice_2:output:0Bsequential_91/lstm_118/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ƒ
<sequential_91/lstm_118/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOpEsequential_91_lstm_118_lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0„
-sequential_91/lstm_118/lstm_cell_123/MatMul_1MatMul%sequential_91/lstm_118/zeros:output:0Dsequential_91/lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–‘
(sequential_91/lstm_118/lstm_cell_123/addAddV25sequential_91/lstm_118/lstm_cell_123/MatMul:product:07sequential_91/lstm_118/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–љ
;sequential_91/lstm_118/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOpDsequential_91_lstm_118_lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ё
,sequential_91/lstm_118/lstm_cell_123/BiasAddBiasAdd,sequential_91/lstm_118/lstm_cell_123/add:z:0Csequential_91/lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–v
4sequential_91/lstm_118/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :©
*sequential_91/lstm_118/lstm_cell_123/splitSplit=sequential_91/lstm_118/lstm_cell_123/split/split_dim:output:05sequential_91/lstm_118/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitЯ
,sequential_91/lstm_118/lstm_cell_123/SigmoidSigmoid3sequential_91/lstm_118/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і°
.sequential_91/lstm_118/lstm_cell_123/Sigmoid_1Sigmoid3sequential_91/lstm_118/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іњ
(sequential_91/lstm_118/lstm_cell_123/mulMul2sequential_91/lstm_118/lstm_cell_123/Sigmoid_1:y:0'sequential_91/lstm_118/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЩ
)sequential_91/lstm_118/lstm_cell_123/ReluRelu3sequential_91/lstm_118/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іѕ
*sequential_91/lstm_118/lstm_cell_123/mul_1Mul0sequential_91/lstm_118/lstm_cell_123/Sigmoid:y:07sequential_91/lstm_118/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іƒ
*sequential_91/lstm_118/lstm_cell_123/add_1AddV2,sequential_91/lstm_118/lstm_cell_123/mul:z:0.sequential_91/lstm_118/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і°
.sequential_91/lstm_118/lstm_cell_123/Sigmoid_2Sigmoid3sequential_91/lstm_118/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЦ
+sequential_91/lstm_118/lstm_cell_123/Relu_1Relu.sequential_91/lstm_118/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і”
*sequential_91/lstm_118/lstm_cell_123/mul_2Mul2sequential_91/lstm_118/lstm_cell_123/Sigmoid_2:y:09sequential_91/lstm_118/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
4sequential_91/lstm_118/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   э
&sequential_91/lstm_118/TensorArrayV2_1TensorListReserve=sequential_91/lstm_118/TensorArrayV2_1/element_shape:output:0/sequential_91/lstm_118/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“]
sequential_91/lstm_118/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_91/lstm_118/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€k
)sequential_91/lstm_118/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
sequential_91/lstm_118/whileWhile2sequential_91/lstm_118/while/loop_counter:output:08sequential_91/lstm_118/while/maximum_iterations:output:0$sequential_91/lstm_118/time:output:0/sequential_91/lstm_118/TensorArrayV2_1:handle:0%sequential_91/lstm_118/zeros:output:0'sequential_91/lstm_118/zeros_1:output:0/sequential_91/lstm_118/strided_slice_1:output:0Nsequential_91/lstm_118/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_91_lstm_118_lstm_cell_123_matmul_readvariableop_resourceEsequential_91_lstm_118_lstm_cell_123_matmul_1_readvariableop_resourceDsequential_91_lstm_118_lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_91_lstm_118_while_body_23344227*6
cond.R,
*sequential_91_lstm_118_while_cond_23344226*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Ш
Gsequential_91/lstm_118/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   И
9sequential_91/lstm_118/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_91/lstm_118/while:output:3Psequential_91/lstm_118/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0
,sequential_91/lstm_118/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€x
.sequential_91/lstm_118/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_118/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
&sequential_91/lstm_118/strided_slice_3StridedSliceBsequential_91/lstm_118/TensorArrayV2Stack/TensorListStack:tensor:05sequential_91/lstm_118/strided_slice_3/stack:output:07sequential_91/lstm_118/strided_slice_3/stack_1:output:07sequential_91/lstm_118/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_mask|
'sequential_91/lstm_118/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          №
"sequential_91/lstm_118/transpose_1	TransposeBsequential_91/lstm_118/TensorArrayV2Stack/TensorListStack:tensor:00sequential_91/lstm_118/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іr
sequential_91/lstm_118/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    r
sequential_91/lstm_119/ShapeShape&sequential_91/lstm_118/transpose_1:y:0*
T0*
_output_shapes
:t
*sequential_91/lstm_119/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_91/lstm_119/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_91/lstm_119/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ƒ
$sequential_91/lstm_119/strided_sliceStridedSlice%sequential_91/lstm_119/Shape:output:03sequential_91/lstm_119/strided_slice/stack:output:05sequential_91/lstm_119/strided_slice/stack_1:output:05sequential_91/lstm_119/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential_91/lstm_119/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іЄ
#sequential_91/lstm_119/zeros/packedPack-sequential_91/lstm_119/strided_slice:output:0.sequential_91/lstm_119/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_91/lstm_119/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ≤
sequential_91/lstm_119/zerosFill,sequential_91/lstm_119/zeros/packed:output:0+sequential_91/lstm_119/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іj
'sequential_91/lstm_119/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іЉ
%sequential_91/lstm_119/zeros_1/packedPack-sequential_91/lstm_119/strided_slice:output:00sequential_91/lstm_119/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:i
$sequential_91/lstm_119/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
sequential_91/lstm_119/zeros_1Fill.sequential_91/lstm_119/zeros_1/packed:output:0-sequential_91/lstm_119/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іz
%sequential_91/lstm_119/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Љ
 sequential_91/lstm_119/transpose	Transpose&sequential_91/lstm_118/transpose_1:y:0.sequential_91/lstm_119/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іr
sequential_91/lstm_119/Shape_1Shape$sequential_91/lstm_119/transpose:y:0*
T0*
_output_shapes
:v
,sequential_91/lstm_119/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_119/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_91/lstm_119/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
&sequential_91/lstm_119/strided_slice_1StridedSlice'sequential_91/lstm_119/Shape_1:output:05sequential_91/lstm_119/strided_slice_1/stack:output:07sequential_91/lstm_119/strided_slice_1/stack_1:output:07sequential_91/lstm_119/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
2sequential_91/lstm_119/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€щ
$sequential_91/lstm_119/TensorArrayV2TensorListReserve;sequential_91/lstm_119/TensorArrayV2/element_shape:output:0/sequential_91/lstm_119/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Э
Lsequential_91/lstm_119/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   •
>sequential_91/lstm_119/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$sequential_91/lstm_119/transpose:y:0Usequential_91/lstm_119/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“v
,sequential_91/lstm_119/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_119/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.sequential_91/lstm_119/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
&sequential_91/lstm_119/strided_slice_2StridedSlice$sequential_91/lstm_119/transpose:y:05sequential_91/lstm_119/strided_slice_2/stack:output:07sequential_91/lstm_119/strided_slice_2/stack_1:output:07sequential_91/lstm_119/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskј
:sequential_91/lstm_119/lstm_cell_124/MatMul/ReadVariableOpReadVariableOpCsequential_91_lstm_119_lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ё
+sequential_91/lstm_119/lstm_cell_124/MatMulMatMul/sequential_91/lstm_119/strided_slice_2:output:0Bsequential_91/lstm_119/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ƒ
<sequential_91/lstm_119/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOpEsequential_91_lstm_119_lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0„
-sequential_91/lstm_119/lstm_cell_124/MatMul_1MatMul%sequential_91/lstm_119/zeros:output:0Dsequential_91/lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–‘
(sequential_91/lstm_119/lstm_cell_124/addAddV25sequential_91/lstm_119/lstm_cell_124/MatMul:product:07sequential_91/lstm_119/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–љ
;sequential_91/lstm_119/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOpDsequential_91_lstm_119_lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ё
,sequential_91/lstm_119/lstm_cell_124/BiasAddBiasAdd,sequential_91/lstm_119/lstm_cell_124/add:z:0Csequential_91/lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–v
4sequential_91/lstm_119/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :©
*sequential_91/lstm_119/lstm_cell_124/splitSplit=sequential_91/lstm_119/lstm_cell_124/split/split_dim:output:05sequential_91/lstm_119/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitЯ
,sequential_91/lstm_119/lstm_cell_124/SigmoidSigmoid3sequential_91/lstm_119/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і°
.sequential_91/lstm_119/lstm_cell_124/Sigmoid_1Sigmoid3sequential_91/lstm_119/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іњ
(sequential_91/lstm_119/lstm_cell_124/mulMul2sequential_91/lstm_119/lstm_cell_124/Sigmoid_1:y:0'sequential_91/lstm_119/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЩ
)sequential_91/lstm_119/lstm_cell_124/ReluRelu3sequential_91/lstm_119/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іѕ
*sequential_91/lstm_119/lstm_cell_124/mul_1Mul0sequential_91/lstm_119/lstm_cell_124/Sigmoid:y:07sequential_91/lstm_119/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іƒ
*sequential_91/lstm_119/lstm_cell_124/add_1AddV2,sequential_91/lstm_119/lstm_cell_124/mul:z:0.sequential_91/lstm_119/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і°
.sequential_91/lstm_119/lstm_cell_124/Sigmoid_2Sigmoid3sequential_91/lstm_119/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЦ
+sequential_91/lstm_119/lstm_cell_124/Relu_1Relu.sequential_91/lstm_119/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і”
*sequential_91/lstm_119/lstm_cell_124/mul_2Mul2sequential_91/lstm_119/lstm_cell_124/Sigmoid_2:y:09sequential_91/lstm_119/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
4sequential_91/lstm_119/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   u
3sequential_91/lstm_119/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :К
&sequential_91/lstm_119/TensorArrayV2_1TensorListReserve=sequential_91/lstm_119/TensorArrayV2_1/element_shape:output:0<sequential_91/lstm_119/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“]
sequential_91/lstm_119/timeConst*
_output_shapes
: *
dtype0*
value	B : z
/sequential_91/lstm_119/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€k
)sequential_91/lstm_119/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѕ
sequential_91/lstm_119/whileWhile2sequential_91/lstm_119/while/loop_counter:output:08sequential_91/lstm_119/while/maximum_iterations:output:0$sequential_91/lstm_119/time:output:0/sequential_91/lstm_119/TensorArrayV2_1:handle:0%sequential_91/lstm_119/zeros:output:0'sequential_91/lstm_119/zeros_1:output:0/sequential_91/lstm_119/strided_slice_1:output:0Nsequential_91/lstm_119/TensorArrayUnstack/TensorListFromTensor:output_handle:0Csequential_91_lstm_119_lstm_cell_124_matmul_readvariableop_resourceEsequential_91_lstm_119_lstm_cell_124_matmul_1_readvariableop_resourceDsequential_91_lstm_119_lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_91_lstm_119_while_body_23344367*6
cond.R,
*sequential_91_lstm_119_while_cond_23344366*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Ш
Gsequential_91/lstm_119/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Ь
9sequential_91/lstm_119/TensorArrayV2Stack/TensorListStackTensorListStack%sequential_91/lstm_119/while:output:3Psequential_91/lstm_119/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0*
num_elements
,sequential_91/lstm_119/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€x
.sequential_91/lstm_119/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.sequential_91/lstm_119/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
&sequential_91/lstm_119/strided_slice_3StridedSliceBsequential_91/lstm_119/TensorArrayV2Stack/TensorListStack:tensor:05sequential_91/lstm_119/strided_slice_3/stack:output:07sequential_91/lstm_119/strided_slice_3/stack_1:output:07sequential_91/lstm_119/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_mask|
'sequential_91/lstm_119/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          №
"sequential_91/lstm_119/transpose_1	TransposeBsequential_91/lstm_119/TensorArrayV2Stack/TensorListStack:tensor:00sequential_91/lstm_119/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іr
sequential_91/lstm_119/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    С
!sequential_91/dropout_72/IdentityIdentity/sequential_91/lstm_119/strided_slice_3:output:0*
T0*(
_output_shapes
:€€€€€€€€€і£
,sequential_91/dense_89/MatMul/ReadVariableOpReadVariableOp5sequential_91_dense_89_matmul_readvariableop_resource*
_output_shapes
:	і*
dtype0ї
sequential_91/dense_89/MatMulMatMul*sequential_91/dropout_72/Identity:output:04sequential_91/dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€†
-sequential_91/dense_89/BiasAdd/ReadVariableOpReadVariableOp6sequential_91_dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ї
sequential_91/dense_89/BiasAddBiasAdd'sequential_91/dense_89/MatMul:product:05sequential_91/dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
IdentityIdentity'sequential_91/dense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€∞
NoOpNoOp.^sequential_91/dense_89/BiasAdd/ReadVariableOp-^sequential_91/dense_89/MatMul/ReadVariableOp<^sequential_91/lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp;^sequential_91/lstm_117/lstm_cell_122/MatMul/ReadVariableOp=^sequential_91/lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp^sequential_91/lstm_117/while<^sequential_91/lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp;^sequential_91/lstm_118/lstm_cell_123/MatMul/ReadVariableOp=^sequential_91/lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp^sequential_91/lstm_118/while<^sequential_91/lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp;^sequential_91/lstm_119/lstm_cell_124/MatMul/ReadVariableOp=^sequential_91/lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp^sequential_91/lstm_119/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 2^
-sequential_91/dense_89/BiasAdd/ReadVariableOp-sequential_91/dense_89/BiasAdd/ReadVariableOp2\
,sequential_91/dense_89/MatMul/ReadVariableOp,sequential_91/dense_89/MatMul/ReadVariableOp2z
;sequential_91/lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp;sequential_91/lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp2x
:sequential_91/lstm_117/lstm_cell_122/MatMul/ReadVariableOp:sequential_91/lstm_117/lstm_cell_122/MatMul/ReadVariableOp2|
<sequential_91/lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp<sequential_91/lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp2<
sequential_91/lstm_117/whilesequential_91/lstm_117/while2z
;sequential_91/lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp;sequential_91/lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp2x
:sequential_91/lstm_118/lstm_cell_123/MatMul/ReadVariableOp:sequential_91/lstm_118/lstm_cell_123/MatMul/ReadVariableOp2|
<sequential_91/lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp<sequential_91/lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp2<
sequential_91/lstm_118/whilesequential_91/lstm_118/while2z
;sequential_91/lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp;sequential_91/lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp2x
:sequential_91/lstm_119/lstm_cell_124/MatMul/ReadVariableOp:sequential_91/lstm_119/lstm_cell_124/MatMul/ReadVariableOp2|
<sequential_91/lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp<sequential_91/lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp2<
sequential_91/lstm_119/whilesequential_91/lstm_119/while:[ W
+
_output_shapes
:€€€€€€€€€
(
_user_specified_namelstm_117_input
т
Й
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345374

inputs

states
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates
√
Ќ
while_cond_23346457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23346457___redundant_placeholder06
2while_while_cond_23346457___redundant_placeholder16
2while_while_cond_23346457___redundant_placeholder26
2while_while_cond_23346457___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
м8
я
while_body_23345728
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_123_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_123_matmul_readvariableop_resource:
і–H
4while_lstm_cell_123_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_123/BiasAdd/ReadVariableOpҐ)while/lstm_cell_123/MatMul/ReadVariableOpҐ+while/lstm_cell_123/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_123/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_123/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_123/addAddV2$while/lstm_cell_123/MatMul:product:0&while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_123/BiasAddBiasAddwhile/lstm_cell_123/add:z:02while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_123/splitSplit,while/lstm_cell_123/split/split_dim:output:0$while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_123/SigmoidSigmoid"while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_1Sigmoid"while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_123/mulMul!while/lstm_cell_123/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_123/ReluRelu"while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_123/mul_1Mulwhile/lstm_cell_123/Sigmoid:y:0&while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_123/add_1AddV2while/lstm_cell_123/mul:z:0while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_2Sigmoid"while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_123/Relu_1Reluwhile/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_123/mul_2Mul!while/lstm_cell_123/Sigmoid_2:y:0(while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_123/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_123/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_123/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_123/BiasAdd/ReadVariableOp*^while/lstm_cell_123/MatMul/ReadVariableOp,^while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_123_biasadd_readvariableop_resource5while_lstm_cell_123_biasadd_readvariableop_resource_0"n
4while_lstm_cell_123_matmul_1_readvariableop_resource6while_lstm_cell_123_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_123_matmul_readvariableop_resource4while_lstm_cell_123_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_123/BiasAdd/ReadVariableOp*while/lstm_cell_123/BiasAdd/ReadVariableOp2V
)while/lstm_cell_123/MatMul/ReadVariableOp)while/lstm_cell_123/MatMul/ReadVariableOp2Z
+while/lstm_cell_123/MatMul_1/ReadVariableOp+while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ћ8
У
F__inference_lstm_117_layer_call_and_return_conditional_losses_23344800

inputs)
lstm_cell_122_23344718:	–*
lstm_cell_122_23344720:
і–%
lstm_cell_122_23344722:	–
identityИҐ%lstm_cell_122/StatefulPartitionedCallҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskЕ
%lstm_cell_122/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_122_23344718lstm_cell_122_23344720lstm_cell_122_23344722*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344672n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_122_23344718lstm_cell_122_23344720lstm_cell_122_23344722*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23344731*
condR
while_cond_23344730*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іv
NoOpNoOp&^lstm_cell_122/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_122/StatefulPartitionedCall%lstm_cell_122/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
√
Ќ
while_cond_23348065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23348065___redundant_placeholder06
2while_while_cond_23348065___redundant_placeholder16
2while_while_cond_23348065___redundant_placeholder26
2while_while_cond_23348065___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
€
ы
0__inference_lstm_cell_124_layer_call_fn_23349792

inputs
states_0
states_1
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identity

identity_1

identity_2ИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345226p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
Ќ	
ш
F__inference_dense_89_layer_call_and_return_conditional_losses_23349579

inputs1
matmul_readvariableop_resource:	і-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	і*
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
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€і: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
и8
Ё
while_body_23345578
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_122_matmul_readvariableop_resource_0:	–J
6while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_122_matmul_readvariableop_resource:	–H
4while_lstm_cell_122_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_122/BiasAdd/ReadVariableOpҐ)while/lstm_cell_122/MatMul/ReadVariableOpҐ+while/lstm_cell_122/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Я
)while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0Љ
while/lstm_cell_122/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_122/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_122/addAddV2$while/lstm_cell_122/MatMul:product:0&while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_122/BiasAddBiasAddwhile/lstm_cell_122/add:z:02while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_122/splitSplit,while/lstm_cell_122/split/split_dim:output:0$while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_122/SigmoidSigmoid"while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_1Sigmoid"while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_122/mulMul!while/lstm_cell_122/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_122/ReluRelu"while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_122/mul_1Mulwhile/lstm_cell_122/Sigmoid:y:0&while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_122/add_1AddV2while/lstm_cell_122/mul:z:0while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_2Sigmoid"while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_122/Relu_1Reluwhile/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_122/mul_2Mul!while/lstm_cell_122/Sigmoid_2:y:0(while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_122/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_122/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_122/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_122/BiasAdd/ReadVariableOp*^while/lstm_cell_122/MatMul/ReadVariableOp,^while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_122_biasadd_readvariableop_resource5while_lstm_cell_122_biasadd_readvariableop_resource_0"n
4while_lstm_cell_122_matmul_1_readvariableop_resource6while_lstm_cell_122_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_122_matmul_readvariableop_resource4while_lstm_cell_122_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_122/BiasAdd/ReadVariableOp*while/lstm_cell_122/BiasAdd/ReadVariableOp2V
)while/lstm_cell_122/MatMul/ReadVariableOp)while/lstm_cell_122/MatMul/ReadVariableOp2Z
+while/lstm_cell_122/MatMul_1/ReadVariableOp+while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
и8
Ё
while_body_23348209
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_122_matmul_readvariableop_resource_0:	–J
6while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_122_matmul_readvariableop_resource:	–H
4while_lstm_cell_122_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_122/BiasAdd/ReadVariableOpҐ)while/lstm_cell_122/MatMul/ReadVariableOpҐ+while/lstm_cell_122/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Я
)while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0Љ
while/lstm_cell_122/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_122/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_122/addAddV2$while/lstm_cell_122/MatMul:product:0&while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_122/BiasAddBiasAddwhile/lstm_cell_122/add:z:02while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_122/splitSplit,while/lstm_cell_122/split/split_dim:output:0$while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_122/SigmoidSigmoid"while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_1Sigmoid"while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_122/mulMul!while/lstm_cell_122/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_122/ReluRelu"while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_122/mul_1Mulwhile/lstm_cell_122/Sigmoid:y:0&while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_122/add_1AddV2while/lstm_cell_122/mul:z:0while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_2Sigmoid"while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_122/Relu_1Reluwhile/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_122/mul_2Mul!while/lstm_cell_122/Sigmoid_2:y:0(while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_122/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_122/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_122/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_122/BiasAdd/ReadVariableOp*^while/lstm_cell_122/MatMul/ReadVariableOp,^while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_122_biasadd_readvariableop_resource5while_lstm_cell_122_biasadd_readvariableop_resource_0"n
4while_lstm_cell_122_matmul_1_readvariableop_resource6while_lstm_cell_122_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_122_matmul_readvariableop_resource4while_lstm_cell_122_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_122/BiasAdd/ReadVariableOp*while/lstm_cell_122/BiasAdd/ReadVariableOp2V
)while/lstm_cell_122/MatMul/ReadVariableOp)while/lstm_cell_122/MatMul/ReadVariableOp2Z
+while/lstm_cell_122/MatMul_1/ReadVariableOp+while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
Љ9
Ф
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345504

inputs*
lstm_cell_124_23345420:
і–*
lstm_cell_124_23345422:
і–%
lstm_cell_124_23345424:	–
identityИҐ%lstm_cell_124/StatefulPartitionedCallҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskЕ
%lstm_cell_124/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_124_23345420lstm_cell_124_23345422lstm_cell_124_23345424*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345374n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_124_23345420lstm_cell_124_23345422lstm_cell_124_23345424*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23345434*
condR
while_cond_23345433*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іv
NoOpNoOp&^lstm_cell_124/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2N
%lstm_cell_124/StatefulPartitionedCall%lstm_cell_124/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
 
_user_specified_nameinputs
√
Ќ
while_cond_23347922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23347922___redundant_placeholder06
2while_while_cond_23347922___redundant_placeholder16
2while_while_cond_23347922___redundant_placeholder26
2while_while_cond_23347922___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
—8
Ф
F__inference_lstm_118_layer_call_and_return_conditional_losses_23344959

inputs*
lstm_cell_123_23344877:
і–*
lstm_cell_123_23344879:
і–%
lstm_cell_123_23344881:	–
identityИҐ%lstm_cell_123/StatefulPartitionedCallҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskЕ
%lstm_cell_123/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_123_23344877lstm_cell_123_23344879lstm_cell_123_23344881*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23344876n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_123_23344877lstm_cell_123_23344879lstm_cell_123_23344881*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23344890*
condR
while_cond_23344889*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іv
NoOpNoOp&^lstm_cell_123/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2N
%lstm_cell_123/StatefulPartitionedCall%lstm_cell_123/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
 
_user_specified_nameinputs
бD
€

lstm_119_while_body_23347578.
*lstm_119_while_lstm_119_while_loop_counter4
0lstm_119_while_lstm_119_while_maximum_iterations
lstm_119_while_placeholder 
lstm_119_while_placeholder_1 
lstm_119_while_placeholder_2 
lstm_119_while_placeholder_3-
)lstm_119_while_lstm_119_strided_slice_1_0i
elstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0:
і–S
?lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–M
>lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
lstm_119_while_identity
lstm_119_while_identity_1
lstm_119_while_identity_2
lstm_119_while_identity_3
lstm_119_while_identity_4
lstm_119_while_identity_5+
'lstm_119_while_lstm_119_strided_slice_1g
clstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensorO
;lstm_119_while_lstm_cell_124_matmul_readvariableop_resource:
і–Q
=lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource:
і–K
<lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpҐ2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOpҐ4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOpС
@lstm_119/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ‘
2lstm_119/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensor_0lstm_119_while_placeholderIlstm_119/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0≤
2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp=lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0„
#lstm_119/while/lstm_cell_124/MatMulMatMul9lstm_119/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ґ
4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp?lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Њ
%lstm_119/while/lstm_cell_124/MatMul_1MatMullstm_119_while_placeholder_2<lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Љ
 lstm_119/while/lstm_cell_124/addAddV2-lstm_119/while/lstm_cell_124/MatMul:product:0/lstm_119/while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–ѓ
3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp>lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0≈
$lstm_119/while/lstm_cell_124/BiasAddBiasAdd$lstm_119/while/lstm_cell_124/add:z:0;lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–n
,lstm_119/while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :С
"lstm_119/while/lstm_cell_124/splitSplit5lstm_119/while/lstm_cell_124/split/split_dim:output:0-lstm_119/while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitП
$lstm_119/while/lstm_cell_124/SigmoidSigmoid+lstm_119/while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_119/while/lstm_cell_124/Sigmoid_1Sigmoid+lstm_119/while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€і§
 lstm_119/while/lstm_cell_124/mulMul*lstm_119/while/lstm_cell_124/Sigmoid_1:y:0lstm_119_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іЙ
!lstm_119/while/lstm_cell_124/ReluRelu+lstm_119/while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЈ
"lstm_119/while/lstm_cell_124/mul_1Mul(lstm_119/while/lstm_cell_124/Sigmoid:y:0/lstm_119/while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іђ
"lstm_119/while/lstm_cell_124/add_1AddV2$lstm_119/while/lstm_cell_124/mul:z:0&lstm_119/while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_119/while/lstm_cell_124/Sigmoid_2Sigmoid+lstm_119/while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЖ
#lstm_119/while/lstm_cell_124/Relu_1Relu&lstm_119/while/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ії
"lstm_119/while/lstm_cell_124/mul_2Mul*lstm_119/while/lstm_cell_124/Sigmoid_2:y:01lstm_119/while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і{
9lstm_119/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Т
3lstm_119/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_119_while_placeholder_1Blstm_119/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_119/while/lstm_cell_124/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“V
lstm_119/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_119/while/addAddV2lstm_119_while_placeholderlstm_119/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_119/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
lstm_119/while/add_1AddV2*lstm_119_while_lstm_119_while_loop_counterlstm_119/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_119/while/IdentityIdentitylstm_119/while/add_1:z:0^lstm_119/while/NoOp*
T0*
_output_shapes
: О
lstm_119/while/Identity_1Identity0lstm_119_while_lstm_119_while_maximum_iterations^lstm_119/while/NoOp*
T0*
_output_shapes
: t
lstm_119/while/Identity_2Identitylstm_119/while/add:z:0^lstm_119/while/NoOp*
T0*
_output_shapes
: °
lstm_119/while/Identity_3IdentityClstm_119/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_119/while/NoOp*
T0*
_output_shapes
: Ц
lstm_119/while/Identity_4Identity&lstm_119/while/lstm_cell_124/mul_2:z:0^lstm_119/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іЦ
lstm_119/while/Identity_5Identity&lstm_119/while/lstm_cell_124/add_1:z:0^lstm_119/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іч
lstm_119/while/NoOpNoOp4^lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp3^lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp5^lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_119_while_identity lstm_119/while/Identity:output:0"?
lstm_119_while_identity_1"lstm_119/while/Identity_1:output:0"?
lstm_119_while_identity_2"lstm_119/while/Identity_2:output:0"?
lstm_119_while_identity_3"lstm_119/while/Identity_3:output:0"?
lstm_119_while_identity_4"lstm_119/while/Identity_4:output:0"?
lstm_119_while_identity_5"lstm_119/while/Identity_5:output:0"T
'lstm_119_while_lstm_119_strided_slice_1)lstm_119_while_lstm_119_strided_slice_1_0"~
<lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource>lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0"А
=lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource?lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0"|
;lstm_119_while_lstm_cell_124_matmul_readvariableop_resource=lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0"ћ
clstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensorelstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2j
3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp2h
2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp2l
4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
√
Ќ
while_cond_23346292
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23346292___redundant_placeholder06
2while_while_cond_23346292___redundant_placeholder16
2while_while_cond_23346292___redundant_placeholder26
2while_while_cond_23346292___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ј

Б
lstm_118_while_cond_23347007.
*lstm_118_while_lstm_118_while_loop_counter4
0lstm_118_while_lstm_118_while_maximum_iterations
lstm_118_while_placeholder 
lstm_118_while_placeholder_1 
lstm_118_while_placeholder_2 
lstm_118_while_placeholder_30
,lstm_118_while_less_lstm_118_strided_slice_1H
Dlstm_118_while_lstm_118_while_cond_23347007___redundant_placeholder0H
Dlstm_118_while_lstm_118_while_cond_23347007___redundant_placeholder1H
Dlstm_118_while_lstm_118_while_cond_23347007___redundant_placeholder2H
Dlstm_118_while_lstm_118_while_cond_23347007___redundant_placeholder3
lstm_118_while_identity
Ж
lstm_118/while/LessLesslstm_118_while_placeholder,lstm_118_while_less_lstm_118_strided_slice_1*
T0*
_output_shapes
: ]
lstm_118/while/IdentityIdentitylstm_118/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_118_while_identity lstm_118/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
€
ы
0__inference_lstm_cell_124_layer_call_fn_23349809

inputs
states_0
states_1
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identity

identity_1

identity_2ИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345374p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
ѓL
®
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349243
inputs_0@
,lstm_cell_124_matmul_readvariableop_resource:
і–B
.lstm_cell_124_matmul_1_readvariableop_resource:
і–<
-lstm_cell_124_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_124/BiasAdd/ReadVariableOpҐ#lstm_cell_124/MatMul/ReadVariableOpҐ%lstm_cell_124/MatMul_1/ReadVariableOpҐwhile=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_124/MatMul/ReadVariableOpReadVariableOp,lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_124/MatMulMatMulstrided_slice_2:output:0+lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_124/MatMul_1MatMulzeros:output:0-lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_124/addAddV2lstm_cell_124/MatMul:product:0 lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_124/BiasAddBiasAddlstm_cell_124/add:z:0,lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_124/splitSplit&lstm_cell_124/split/split_dim:output:0lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_124/SigmoidSigmoidlstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_1Sigmoidlstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_124/mulMullstm_cell_124/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_124/ReluRelulstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_124/mul_1Mullstm_cell_124/Sigmoid:y:0 lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_124/add_1AddV2lstm_cell_124/mul:z:0lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_2Sigmoidlstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_124/Relu_1Relulstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_124/mul_2Mullstm_cell_124/Sigmoid_2:y:0"lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_124_matmul_readvariableop_resource.lstm_cell_124_matmul_1_readvariableop_resource-lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23349158*
condR
while_cond_23349157*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_124/BiasAdd/ReadVariableOp$^lstm_cell_124/MatMul/ReadVariableOp&^lstm_cell_124/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2L
$lstm_cell_124/BiasAdd/ReadVariableOp$lstm_cell_124/BiasAdd/ReadVariableOp2J
#lstm_cell_124/MatMul/ReadVariableOp#lstm_cell_124/MatMul/ReadVariableOp2N
%lstm_cell_124/MatMul_1/ReadVariableOp%lstm_cell_124/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
ц
К
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349677

inputs
states_0
states_11
matmul_readvariableop_resource:	–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
ј

Б
lstm_119_while_cond_23347577.
*lstm_119_while_lstm_119_while_loop_counter4
0lstm_119_while_lstm_119_while_maximum_iterations
lstm_119_while_placeholder 
lstm_119_while_placeholder_1 
lstm_119_while_placeholder_2 
lstm_119_while_placeholder_30
,lstm_119_while_less_lstm_119_strided_slice_1H
Dlstm_119_while_lstm_119_while_cond_23347577___redundant_placeholder0H
Dlstm_119_while_lstm_119_while_cond_23347577___redundant_placeholder1H
Dlstm_119_while_lstm_119_while_cond_23347577___redundant_placeholder2H
Dlstm_119_while_lstm_119_while_cond_23347577___redundant_placeholder3
lstm_119_while_identity
Ж
lstm_119/while/LessLesslstm_119_while_placeholder,lstm_119_while_less_lstm_119_strided_slice_1*
T0*
_output_shapes
: ]
lstm_119/while/IdentityIdentitylstm_119/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_119_while_identity lstm_119/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ъ
Л
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349743

inputs
states_0
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
ѓL
®
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349098
inputs_0@
,lstm_cell_124_matmul_readvariableop_resource:
і–B
.lstm_cell_124_matmul_1_readvariableop_resource:
і–<
-lstm_cell_124_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_124/BiasAdd/ReadVariableOpҐ#lstm_cell_124/MatMul/ReadVariableOpҐ%lstm_cell_124/MatMul_1/ReadVariableOpҐwhile=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_124/MatMul/ReadVariableOpReadVariableOp,lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_124/MatMulMatMulstrided_slice_2:output:0+lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_124/MatMul_1MatMulzeros:output:0-lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_124/addAddV2lstm_cell_124/MatMul:product:0 lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_124/BiasAddBiasAddlstm_cell_124/add:z:0,lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_124/splitSplit&lstm_cell_124/split/split_dim:output:0lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_124/SigmoidSigmoidlstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_1Sigmoidlstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_124/mulMullstm_cell_124/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_124/ReluRelulstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_124/mul_1Mullstm_cell_124/Sigmoid:y:0 lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_124/add_1AddV2lstm_cell_124/mul:z:0lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_2Sigmoidlstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_124/Relu_1Relulstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_124/mul_2Mullstm_cell_124/Sigmoid_2:y:0"lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_124_matmul_readvariableop_resource.lstm_cell_124_matmul_1_readvariableop_resource-lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23349013*
condR
while_cond_23349012*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_124/BiasAdd/ReadVariableOp$^lstm_cell_124/MatMul/ReadVariableOp&^lstm_cell_124/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2L
$lstm_cell_124/BiasAdd/ReadVariableOp$lstm_cell_124/BiasAdd/ReadVariableOp2J
#lstm_cell_124/MatMul/ReadVariableOp#lstm_cell_124/MatMul/ReadVariableOp2N
%lstm_cell_124/MatMul_1/ReadVariableOp%lstm_cell_124/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
м8
я
while_body_23348396
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_123_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_123_matmul_readvariableop_resource:
і–H
4while_lstm_cell_123_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_123/BiasAdd/ReadVariableOpҐ)while/lstm_cell_123/MatMul/ReadVariableOpҐ+while/lstm_cell_123/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_123/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_123/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_123/addAddV2$while/lstm_cell_123/MatMul:product:0&while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_123/BiasAddBiasAddwhile/lstm_cell_123/add:z:02while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_123/splitSplit,while/lstm_cell_123/split/split_dim:output:0$while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_123/SigmoidSigmoid"while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_1Sigmoid"while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_123/mulMul!while/lstm_cell_123/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_123/ReluRelu"while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_123/mul_1Mulwhile/lstm_cell_123/Sigmoid:y:0&while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_123/add_1AddV2while/lstm_cell_123/mul:z:0while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_2Sigmoid"while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_123/Relu_1Reluwhile/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_123/mul_2Mul!while/lstm_cell_123/Sigmoid_2:y:0(while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_123/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_123/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_123/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_123/BiasAdd/ReadVariableOp*^while/lstm_cell_123/MatMul/ReadVariableOp,^while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_123_biasadd_readvariableop_resource5while_lstm_cell_123_biasadd_readvariableop_resource_0"n
4while_lstm_cell_123_matmul_1_readvariableop_resource6while_lstm_cell_123_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_123_matmul_readvariableop_resource4while_lstm_cell_123_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_123/BiasAdd/ReadVariableOp*while/lstm_cell_123/BiasAdd/ReadVariableOp2V
)while/lstm_cell_123/MatMul/ReadVariableOp)while/lstm_cell_123/MatMul/ReadVariableOp2Z
+while/lstm_cell_123/MatMul_1/ReadVariableOp+while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ћ8
У
F__inference_lstm_117_layer_call_and_return_conditional_losses_23344609

inputs)
lstm_cell_122_23344527:	–*
lstm_cell_122_23344529:
і–%
lstm_cell_122_23344531:	–
identityИҐ%lstm_cell_122/StatefulPartitionedCallҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskЕ
%lstm_cell_122/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_122_23344527lstm_cell_122_23344529lstm_cell_122_23344531*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344526n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_122_23344527lstm_cell_122_23344529lstm_cell_122_23344531*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23344540*
condR
while_cond_23344539*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іv
NoOpNoOp&^lstm_cell_122/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2N
%lstm_cell_122/StatefulPartitionedCall%lstm_cell_122/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
€

≠
0__inference_sequential_91_layer_call_fn_23346663
lstm_117_input
unknown:	–
	unknown_0:
і–
	unknown_1:	–
	unknown_2:
і–
	unknown_3:
і–
	unknown_4:	–
	unknown_5:
і–
	unknown_6:
і–
	unknown_7:	–
	unknown_8:	і
	unknown_9:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346611o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€
(
_user_specified_namelstm_117_input
√
Ќ
while_cond_23345727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23345727___redundant_placeholder06
2while_while_cond_23345727___redundant_placeholder16
2while_while_cond_23345727___redundant_placeholder26
2while_while_cond_23345727___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
»
Щ
*sequential_91_lstm_117_while_cond_23344087J
Fsequential_91_lstm_117_while_sequential_91_lstm_117_while_loop_counterP
Lsequential_91_lstm_117_while_sequential_91_lstm_117_while_maximum_iterations,
(sequential_91_lstm_117_while_placeholder.
*sequential_91_lstm_117_while_placeholder_1.
*sequential_91_lstm_117_while_placeholder_2.
*sequential_91_lstm_117_while_placeholder_3L
Hsequential_91_lstm_117_while_less_sequential_91_lstm_117_strided_slice_1d
`sequential_91_lstm_117_while_sequential_91_lstm_117_while_cond_23344087___redundant_placeholder0d
`sequential_91_lstm_117_while_sequential_91_lstm_117_while_cond_23344087___redundant_placeholder1d
`sequential_91_lstm_117_while_sequential_91_lstm_117_while_cond_23344087___redundant_placeholder2d
`sequential_91_lstm_117_while_sequential_91_lstm_117_while_cond_23344087___redundant_placeholder3)
%sequential_91_lstm_117_while_identity
Њ
!sequential_91/lstm_117/while/LessLess(sequential_91_lstm_117_while_placeholderHsequential_91_lstm_117_while_less_sequential_91_lstm_117_strided_slice_1*
T0*
_output_shapes
: y
%sequential_91/lstm_117/while/IdentityIdentity%sequential_91/lstm_117/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_91_lstm_117_while_identity.sequential_91/lstm_117/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
€

≠
0__inference_sequential_91_layer_call_fn_23346021
lstm_117_input
unknown:	–
	unknown_0:
і–
	unknown_1:	–
	unknown_2:
і–
	unknown_3:
і–
	unknown_4:	–
	unknown_5:
і–
	unknown_6:
і–
	unknown_7:	–
	unknown_8:	і
	unknown_9:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_91_layer_call_and_return_conditional_losses_23345996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€
(
_user_specified_namelstm_117_input
Ы
Љ
+__inference_lstm_119_layer_call_fn_23348920
inputs_0
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345311p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
Р
є
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346725
lstm_117_input$
lstm_117_23346697:	–%
lstm_117_23346699:
і– 
lstm_117_23346701:	–%
lstm_118_23346704:
і–%
lstm_118_23346706:
і– 
lstm_118_23346708:	–%
lstm_119_23346711:
і–%
lstm_119_23346713:
і– 
lstm_119_23346715:	–$
dense_89_23346719:	і
dense_89_23346721:
identityИҐ dense_89/StatefulPartitionedCallҐ"dropout_72/StatefulPartitionedCallҐ lstm_117/StatefulPartitionedCallҐ lstm_118/StatefulPartitionedCallҐ lstm_119/StatefulPartitionedCallШ
 lstm_117/StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputlstm_117_23346697lstm_117_23346699lstm_117_23346701*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23346542≥
 lstm_118/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0lstm_118_23346704lstm_118_23346706lstm_118_23346708*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23346377ѓ
 lstm_119/StatefulPartitionedCallStatefulPartitionedCall)lstm_118/StatefulPartitionedCall:output:0lstm_119_23346711lstm_119_23346713lstm_119_23346715*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23346212т
"dropout_72/StatefulPartitionedCallStatefulPartitionedCall)lstm_119/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_72_layer_call_and_return_conditional_losses_23346051Ы
 dense_89/StatefulPartitionedCallStatefulPartitionedCall+dropout_72/StatefulPartitionedCall:output:0dense_89_23346719dense_89_23346721*
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
F__inference_dense_89_layer_call_and_return_conditional_losses_23345989x
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ч
NoOpNoOp!^dense_89/StatefulPartitionedCall#^dropout_72/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall!^lstm_118/StatefulPartitionedCall!^lstm_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2H
"dropout_72/StatefulPartitionedCall"dropout_72/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall2D
 lstm_118/StatefulPartitionedCall lstm_118/StatefulPartitionedCall2D
 lstm_119/StatefulPartitionedCall lstm_119/StatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€
(
_user_specified_namelstm_117_input
бD
€

lstm_119_while_body_23347148.
*lstm_119_while_lstm_119_while_loop_counter4
0lstm_119_while_lstm_119_while_maximum_iterations
lstm_119_while_placeholder 
lstm_119_while_placeholder_1 
lstm_119_while_placeholder_2 
lstm_119_while_placeholder_3-
)lstm_119_while_lstm_119_strided_slice_1_0i
elstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0:
і–S
?lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–M
>lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
lstm_119_while_identity
lstm_119_while_identity_1
lstm_119_while_identity_2
lstm_119_while_identity_3
lstm_119_while_identity_4
lstm_119_while_identity_5+
'lstm_119_while_lstm_119_strided_slice_1g
clstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensorO
;lstm_119_while_lstm_cell_124_matmul_readvariableop_resource:
і–Q
=lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource:
і–K
<lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpҐ2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOpҐ4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOpС
@lstm_119/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ‘
2lstm_119/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensor_0lstm_119_while_placeholderIlstm_119/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0≤
2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp=lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0„
#lstm_119/while/lstm_cell_124/MatMulMatMul9lstm_119/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ґ
4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp?lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Њ
%lstm_119/while/lstm_cell_124/MatMul_1MatMullstm_119_while_placeholder_2<lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Љ
 lstm_119/while/lstm_cell_124/addAddV2-lstm_119/while/lstm_cell_124/MatMul:product:0/lstm_119/while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–ѓ
3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp>lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0≈
$lstm_119/while/lstm_cell_124/BiasAddBiasAdd$lstm_119/while/lstm_cell_124/add:z:0;lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–n
,lstm_119/while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :С
"lstm_119/while/lstm_cell_124/splitSplit5lstm_119/while/lstm_cell_124/split/split_dim:output:0-lstm_119/while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitП
$lstm_119/while/lstm_cell_124/SigmoidSigmoid+lstm_119/while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_119/while/lstm_cell_124/Sigmoid_1Sigmoid+lstm_119/while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€і§
 lstm_119/while/lstm_cell_124/mulMul*lstm_119/while/lstm_cell_124/Sigmoid_1:y:0lstm_119_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іЙ
!lstm_119/while/lstm_cell_124/ReluRelu+lstm_119/while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЈ
"lstm_119/while/lstm_cell_124/mul_1Mul(lstm_119/while/lstm_cell_124/Sigmoid:y:0/lstm_119/while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іђ
"lstm_119/while/lstm_cell_124/add_1AddV2$lstm_119/while/lstm_cell_124/mul:z:0&lstm_119/while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_119/while/lstm_cell_124/Sigmoid_2Sigmoid+lstm_119/while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЖ
#lstm_119/while/lstm_cell_124/Relu_1Relu&lstm_119/while/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ії
"lstm_119/while/lstm_cell_124/mul_2Mul*lstm_119/while/lstm_cell_124/Sigmoid_2:y:01lstm_119/while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і{
9lstm_119/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Т
3lstm_119/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_119_while_placeholder_1Blstm_119/while/TensorArrayV2Write/TensorListSetItem/index:output:0&lstm_119/while/lstm_cell_124/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“V
lstm_119/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_119/while/addAddV2lstm_119_while_placeholderlstm_119/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_119/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
lstm_119/while/add_1AddV2*lstm_119_while_lstm_119_while_loop_counterlstm_119/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_119/while/IdentityIdentitylstm_119/while/add_1:z:0^lstm_119/while/NoOp*
T0*
_output_shapes
: О
lstm_119/while/Identity_1Identity0lstm_119_while_lstm_119_while_maximum_iterations^lstm_119/while/NoOp*
T0*
_output_shapes
: t
lstm_119/while/Identity_2Identitylstm_119/while/add:z:0^lstm_119/while/NoOp*
T0*
_output_shapes
: °
lstm_119/while/Identity_3IdentityClstm_119/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_119/while/NoOp*
T0*
_output_shapes
: Ц
lstm_119/while/Identity_4Identity&lstm_119/while/lstm_cell_124/mul_2:z:0^lstm_119/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іЦ
lstm_119/while/Identity_5Identity&lstm_119/while/lstm_cell_124/add_1:z:0^lstm_119/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іч
lstm_119/while/NoOpNoOp4^lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp3^lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp5^lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_119_while_identity lstm_119/while/Identity:output:0"?
lstm_119_while_identity_1"lstm_119/while/Identity_1:output:0"?
lstm_119_while_identity_2"lstm_119/while/Identity_2:output:0"?
lstm_119_while_identity_3"lstm_119/while/Identity_3:output:0"?
lstm_119_while_identity_4"lstm_119/while/Identity_4:output:0"?
lstm_119_while_identity_5"lstm_119/while/Identity_5:output:0"T
'lstm_119_while_lstm_119_strided_slice_1)lstm_119_while_lstm_119_strided_slice_1_0"~
<lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource>lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0"А
=lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource?lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0"|
;lstm_119_while_lstm_cell_124_matmul_readvariableop_resource=lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0"ћ
clstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensorelstm_119_while_tensorarrayv2read_tensorlistgetitem_lstm_119_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2j
3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp3lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp2h
2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp2lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp2l
4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp4lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ЖK
¶
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348766

inputs@
,lstm_cell_123_matmul_readvariableop_resource:
і–B
.lstm_cell_123_matmul_1_readvariableop_resource:
і–<
-lstm_cell_123_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_123/BiasAdd/ReadVariableOpҐ#lstm_cell_123/MatMul/ReadVariableOpҐ%lstm_cell_123/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_123/MatMul/ReadVariableOpReadVariableOp,lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_123/MatMulMatMulstrided_slice_2:output:0+lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_123/MatMul_1MatMulzeros:output:0-lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_123/addAddV2lstm_cell_123/MatMul:product:0 lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_123/BiasAddBiasAddlstm_cell_123/add:z:0,lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_123/splitSplit&lstm_cell_123/split/split_dim:output:0lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_123/SigmoidSigmoidlstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_1Sigmoidlstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_123/mulMullstm_cell_123/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_123/ReluRelulstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_123/mul_1Mullstm_cell_123/Sigmoid:y:0 lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_123/add_1AddV2lstm_cell_123/mul:z:0lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_2Sigmoidlstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_123/Relu_1Relulstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_123/mul_2Mullstm_cell_123/Sigmoid_2:y:0"lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_123_matmul_readvariableop_resource.lstm_cell_123_matmul_1_readvariableop_resource-lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23348682*
condR
while_cond_23348681*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_123/BiasAdd/ReadVariableOp$^lstm_cell_123/MatMul/ReadVariableOp&^lstm_cell_123/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_123/BiasAdd/ReadVariableOp$lstm_cell_123/BiasAdd/ReadVariableOp2J
#lstm_cell_123/MatMul/ReadVariableOp#lstm_cell_123/MatMul/ReadVariableOp2N
%lstm_cell_123/MatMul_1/ReadVariableOp%lstm_cell_123/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
и8
Ё
while_body_23348066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_122_matmul_readvariableop_resource_0:	–J
6while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_122_matmul_readvariableop_resource:	–H
4while_lstm_cell_122_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_122/BiasAdd/ReadVariableOpҐ)while/lstm_cell_122/MatMul/ReadVariableOpҐ+while/lstm_cell_122/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Я
)while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0Љ
while/lstm_cell_122/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_122/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_122/addAddV2$while/lstm_cell_122/MatMul:product:0&while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_122/BiasAddBiasAddwhile/lstm_cell_122/add:z:02while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_122/splitSplit,while/lstm_cell_122/split/split_dim:output:0$while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_122/SigmoidSigmoid"while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_1Sigmoid"while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_122/mulMul!while/lstm_cell_122/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_122/ReluRelu"while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_122/mul_1Mulwhile/lstm_cell_122/Sigmoid:y:0&while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_122/add_1AddV2while/lstm_cell_122/mul:z:0while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_2Sigmoid"while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_122/Relu_1Reluwhile/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_122/mul_2Mul!while/lstm_cell_122/Sigmoid_2:y:0(while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_122/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_122/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_122/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_122/BiasAdd/ReadVariableOp*^while/lstm_cell_122/MatMul/ReadVariableOp,^while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_122_biasadd_readvariableop_resource5while_lstm_cell_122_biasadd_readvariableop_resource_0"n
4while_lstm_cell_122_matmul_1_readvariableop_resource6while_lstm_cell_122_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_122_matmul_readvariableop_resource4while_lstm_cell_122_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_122/BiasAdd/ReadVariableOp*while/lstm_cell_122/BiasAdd/ReadVariableOp2V
)while/lstm_cell_122/MatMul/ReadVariableOp)while/lstm_cell_122/MatMul/ReadVariableOp2Z
+while/lstm_cell_122/MatMul_1/ReadVariableOp+while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ј

Б
lstm_118_while_cond_23347437.
*lstm_118_while_lstm_118_while_loop_counter4
0lstm_118_while_lstm_118_while_maximum_iterations
lstm_118_while_placeholder 
lstm_118_while_placeholder_1 
lstm_118_while_placeholder_2 
lstm_118_while_placeholder_30
,lstm_118_while_less_lstm_118_strided_slice_1H
Dlstm_118_while_lstm_118_while_cond_23347437___redundant_placeholder0H
Dlstm_118_while_lstm_118_while_cond_23347437___redundant_placeholder1H
Dlstm_118_while_lstm_118_while_cond_23347437___redundant_placeholder2H
Dlstm_118_while_lstm_118_while_cond_23347437___redundant_placeholder3
lstm_118_while_identity
Ж
lstm_118/while/LessLesslstm_118_while_placeholder,lstm_118_while_less_lstm_118_strided_slice_1*
T0*
_output_shapes
: ]
lstm_118/while/IdentityIdentitylstm_118/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_118_while_identity lstm_118/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
И:
я
while_body_23349448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_124_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_124_matmul_readvariableop_resource:
і–H
4while_lstm_cell_124_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_124/BiasAdd/ReadVariableOpҐ)while/lstm_cell_124/MatMul/ReadVariableOpҐ+while/lstm_cell_124/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_124/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_124/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_124/addAddV2$while/lstm_cell_124/MatMul:product:0&while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_124/BiasAddBiasAddwhile/lstm_cell_124/add:z:02while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_124/splitSplit,while/lstm_cell_124/split/split_dim:output:0$while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_124/SigmoidSigmoid"while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_1Sigmoid"while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_124/mulMul!while/lstm_cell_124/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_124/ReluRelu"while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_124/mul_1Mulwhile/lstm_cell_124/Sigmoid:y:0&while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_124/add_1AddV2while/lstm_cell_124/mul:z:0while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_2Sigmoid"while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_124/Relu_1Reluwhile/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_124/mul_2Mul!while/lstm_cell_124/Sigmoid_2:y:0(while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_124/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_124/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_124/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_124/BiasAdd/ReadVariableOp*^while/lstm_cell_124/MatMul/ReadVariableOp,^while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_124_biasadd_readvariableop_resource5while_lstm_cell_124_biasadd_readvariableop_resource_0"n
4while_lstm_cell_124_matmul_1_readvariableop_resource6while_lstm_cell_124_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_124_matmul_readvariableop_resource4while_lstm_cell_124_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_124/BiasAdd/ReadVariableOp*while/lstm_cell_124/BiasAdd/ReadVariableOp2V
)while/lstm_cell_124/MatMul/ReadVariableOp)while/lstm_cell_124/MatMul/ReadVariableOp2Z
+while/lstm_cell_124/MatMul_1/ReadVariableOp+while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
Љ9
Ф
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345311

inputs*
lstm_cell_124_23345227:
і–*
lstm_cell_124_23345229:
і–%
lstm_cell_124_23345231:	–
identityИҐ%lstm_cell_124/StatefulPartitionedCallҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskЕ
%lstm_cell_124/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_124_23345227lstm_cell_124_23345229lstm_cell_124_23345231*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345226n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_124_23345227lstm_cell_124_23345229lstm_cell_124_23345231*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23345241*
condR
while_cond_23345240*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іv
NoOpNoOp&^lstm_cell_124/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2N
%lstm_cell_124/StatefulPartitionedCall%lstm_cell_124/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
 
_user_specified_nameinputs
ЖK
¶
F__inference_lstm_118_layer_call_and_return_conditional_losses_23345812

inputs@
,lstm_cell_123_matmul_readvariableop_resource:
і–B
.lstm_cell_123_matmul_1_readvariableop_resource:
і–<
-lstm_cell_123_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_123/BiasAdd/ReadVariableOpҐ#lstm_cell_123/MatMul/ReadVariableOpҐ%lstm_cell_123/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_123/MatMul/ReadVariableOpReadVariableOp,lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_123/MatMulMatMulstrided_slice_2:output:0+lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_123/MatMul_1MatMulzeros:output:0-lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_123/addAddV2lstm_cell_123/MatMul:product:0 lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_123/BiasAddBiasAddlstm_cell_123/add:z:0,lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_123/splitSplit&lstm_cell_123/split/split_dim:output:0lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_123/SigmoidSigmoidlstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_1Sigmoidlstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_123/mulMullstm_cell_123/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_123/ReluRelulstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_123/mul_1Mullstm_cell_123/Sigmoid:y:0 lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_123/add_1AddV2lstm_cell_123/mul:z:0lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_2Sigmoidlstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_123/Relu_1Relulstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_123/mul_2Mullstm_cell_123/Sigmoid_2:y:0"lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_123_matmul_readvariableop_resource.lstm_cell_123_matmul_1_readvariableop_resource-lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23345728*
condR
while_cond_23345727*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_123/BiasAdd/ReadVariableOp$^lstm_cell_123/MatMul/ReadVariableOp&^lstm_cell_123/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_123/BiasAdd/ReadVariableOp$lstm_cell_123/BiasAdd/ReadVariableOp2J
#lstm_cell_123/MatMul/ReadVariableOp#lstm_cell_123/MatMul/ReadVariableOp2N
%lstm_cell_123/MatMul_1/ReadVariableOp%lstm_cell_123/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
ќД
†
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347240

inputsH
5lstm_117_lstm_cell_122_matmul_readvariableop_resource:	–K
7lstm_117_lstm_cell_122_matmul_1_readvariableop_resource:
і–E
6lstm_117_lstm_cell_122_biasadd_readvariableop_resource:	–I
5lstm_118_lstm_cell_123_matmul_readvariableop_resource:
і–K
7lstm_118_lstm_cell_123_matmul_1_readvariableop_resource:
і–E
6lstm_118_lstm_cell_123_biasadd_readvariableop_resource:	–I
5lstm_119_lstm_cell_124_matmul_readvariableop_resource:
і–K
7lstm_119_lstm_cell_124_matmul_1_readvariableop_resource:
і–E
6lstm_119_lstm_cell_124_biasadd_readvariableop_resource:	–:
'dense_89_matmul_readvariableop_resource:	і6
(dense_89_biasadd_readvariableop_resource:
identityИҐdense_89/BiasAdd/ReadVariableOpҐdense_89/MatMul/ReadVariableOpҐ-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOpҐ,lstm_117/lstm_cell_122/MatMul/ReadVariableOpҐ.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOpҐlstm_117/whileҐ-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOpҐ,lstm_118/lstm_cell_123/MatMul/ReadVariableOpҐ.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOpҐlstm_118/whileҐ-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOpҐ,lstm_119/lstm_cell_124/MatMul/ReadVariableOpҐ.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOpҐlstm_119/whileD
lstm_117/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_117/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_117/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_117/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
lstm_117/strided_sliceStridedSlicelstm_117/Shape:output:0%lstm_117/strided_slice/stack:output:0'lstm_117/strided_slice/stack_1:output:0'lstm_117/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
lstm_117/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іО
lstm_117/zeros/packedPacklstm_117/strided_slice:output:0 lstm_117/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_117/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
lstm_117/zerosFilllstm_117/zeros/packed:output:0lstm_117/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€і\
lstm_117/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іТ
lstm_117/zeros_1/packedPacklstm_117/strided_slice:output:0"lstm_117/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_117/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    О
lstm_117/zeros_1Fill lstm_117/zeros_1/packed:output:0lstm_117/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іl
lstm_117/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_117/transpose	Transposeinputs lstm_117/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€V
lstm_117/Shape_1Shapelstm_117/transpose:y:0*
T0*
_output_shapes
:h
lstm_117/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_117/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_117/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
lstm_117/strided_slice_1StridedSlicelstm_117/Shape_1:output:0'lstm_117/strided_slice_1/stack:output:0)lstm_117/strided_slice_1/stack_1:output:0)lstm_117/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_117/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
lstm_117/TensorArrayV2TensorListReserve-lstm_117/TensorArrayV2/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“П
>lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ы
0lstm_117/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_117/transpose:y:0Glstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“h
lstm_117/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_117/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_117/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
lstm_117/strided_slice_2StridedSlicelstm_117/transpose:y:0'lstm_117/strided_slice_2/stack:output:0)lstm_117/strided_slice_2/stack_1:output:0)lstm_117/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask£
,lstm_117/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp5lstm_117_lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0≥
lstm_117/lstm_cell_122/MatMulMatMul!lstm_117/strided_slice_2:output:04lstm_117/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–®
.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp7lstm_117_lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≠
lstm_117/lstm_cell_122/MatMul_1MatMullstm_117/zeros:output:06lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–™
lstm_117/lstm_cell_122/addAddV2'lstm_117/lstm_cell_122/MatMul:product:0)lstm_117/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–°
-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp6lstm_117_lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0≥
lstm_117/lstm_cell_122/BiasAddBiasAddlstm_117/lstm_cell_122/add:z:05lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–h
&lstm_117/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :€
lstm_117/lstm_cell_122/splitSplit/lstm_117/lstm_cell_122/split/split_dim:output:0'lstm_117/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitГ
lstm_117/lstm_cell_122/SigmoidSigmoid%lstm_117/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_117/lstm_cell_122/Sigmoid_1Sigmoid%lstm_117/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іХ
lstm_117/lstm_cell_122/mulMul$lstm_117/lstm_cell_122/Sigmoid_1:y:0lstm_117/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€і}
lstm_117/lstm_cell_122/ReluRelu%lstm_117/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€і•
lstm_117/lstm_cell_122/mul_1Mul"lstm_117/lstm_cell_122/Sigmoid:y:0)lstm_117/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЪ
lstm_117/lstm_cell_122/add_1AddV2lstm_117/lstm_cell_122/mul:z:0 lstm_117/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_117/lstm_cell_122/Sigmoid_2Sigmoid%lstm_117/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_117/lstm_cell_122/Relu_1Relu lstm_117/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і©
lstm_117/lstm_cell_122/mul_2Mul$lstm_117/lstm_cell_122/Sigmoid_2:y:0+lstm_117/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іw
&lstm_117/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ”
lstm_117/TensorArrayV2_1TensorListReserve/lstm_117/TensorArrayV2_1/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“O
lstm_117/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_117/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
lstm_117/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Л
lstm_117/whileWhile$lstm_117/while/loop_counter:output:0*lstm_117/while/maximum_iterations:output:0lstm_117/time:output:0!lstm_117/TensorArrayV2_1:handle:0lstm_117/zeros:output:0lstm_117/zeros_1:output:0!lstm_117/strided_slice_1:output:0@lstm_117/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_117_lstm_cell_122_matmul_readvariableop_resource7lstm_117_lstm_cell_122_matmul_1_readvariableop_resource6lstm_117_lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_117_while_body_23346869*(
cond R
lstm_117_while_cond_23346868*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations К
9lstm_117/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ё
+lstm_117/TensorArrayV2Stack/TensorListStackTensorListStacklstm_117/while:output:3Blstm_117/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0q
lstm_117/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 lstm_117/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_117/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
lstm_117/strided_slice_3StridedSlice4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_117/strided_slice_3/stack:output:0)lstm_117/strided_slice_3/stack_1:output:0)lstm_117/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskn
lstm_117/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ≤
lstm_117/transpose_1	Transpose4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_117/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іd
lstm_117/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_118/ShapeShapelstm_117/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_118/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_118/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_118/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
lstm_118/strided_sliceStridedSlicelstm_118/Shape:output:0%lstm_118/strided_slice/stack:output:0'lstm_118/strided_slice/stack_1:output:0'lstm_118/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
lstm_118/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іО
lstm_118/zeros/packedPacklstm_118/strided_slice:output:0 lstm_118/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_118/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
lstm_118/zerosFilllstm_118/zeros/packed:output:0lstm_118/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€і\
lstm_118/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іТ
lstm_118/zeros_1/packedPacklstm_118/strided_slice:output:0"lstm_118/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_118/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    О
lstm_118/zeros_1Fill lstm_118/zeros_1/packed:output:0lstm_118/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іl
lstm_118/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Т
lstm_118/transpose	Transposelstm_117/transpose_1:y:0 lstm_118/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іV
lstm_118/Shape_1Shapelstm_118/transpose:y:0*
T0*
_output_shapes
:h
lstm_118/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_118/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_118/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
lstm_118/strided_slice_1StridedSlicelstm_118/Shape_1:output:0'lstm_118/strided_slice_1/stack:output:0)lstm_118/strided_slice_1/stack_1:output:0)lstm_118/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_118/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
lstm_118/TensorArrayV2TensorListReserve-lstm_118/TensorArrayV2/element_shape:output:0!lstm_118/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“П
>lstm_118/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ы
0lstm_118/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_118/transpose:y:0Glstm_118/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“h
lstm_118/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_118/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_118/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
lstm_118/strided_slice_2StridedSlicelstm_118/transpose:y:0'lstm_118/strided_slice_2/stack:output:0)lstm_118/strided_slice_2/stack_1:output:0)lstm_118/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_mask§
,lstm_118/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp5lstm_118_lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≥
lstm_118/lstm_cell_123/MatMulMatMul!lstm_118/strided_slice_2:output:04lstm_118/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–®
.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp7lstm_118_lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≠
lstm_118/lstm_cell_123/MatMul_1MatMullstm_118/zeros:output:06lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–™
lstm_118/lstm_cell_123/addAddV2'lstm_118/lstm_cell_123/MatMul:product:0)lstm_118/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–°
-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp6lstm_118_lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0≥
lstm_118/lstm_cell_123/BiasAddBiasAddlstm_118/lstm_cell_123/add:z:05lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–h
&lstm_118/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :€
lstm_118/lstm_cell_123/splitSplit/lstm_118/lstm_cell_123/split/split_dim:output:0'lstm_118/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitГ
lstm_118/lstm_cell_123/SigmoidSigmoid%lstm_118/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_118/lstm_cell_123/Sigmoid_1Sigmoid%lstm_118/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іХ
lstm_118/lstm_cell_123/mulMul$lstm_118/lstm_cell_123/Sigmoid_1:y:0lstm_118/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€і}
lstm_118/lstm_cell_123/ReluRelu%lstm_118/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€і•
lstm_118/lstm_cell_123/mul_1Mul"lstm_118/lstm_cell_123/Sigmoid:y:0)lstm_118/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЪ
lstm_118/lstm_cell_123/add_1AddV2lstm_118/lstm_cell_123/mul:z:0 lstm_118/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_118/lstm_cell_123/Sigmoid_2Sigmoid%lstm_118/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_118/lstm_cell_123/Relu_1Relu lstm_118/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і©
lstm_118/lstm_cell_123/mul_2Mul$lstm_118/lstm_cell_123/Sigmoid_2:y:0+lstm_118/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іw
&lstm_118/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ”
lstm_118/TensorArrayV2_1TensorListReserve/lstm_118/TensorArrayV2_1/element_shape:output:0!lstm_118/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“O
lstm_118/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_118/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
lstm_118/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Л
lstm_118/whileWhile$lstm_118/while/loop_counter:output:0*lstm_118/while/maximum_iterations:output:0lstm_118/time:output:0!lstm_118/TensorArrayV2_1:handle:0lstm_118/zeros:output:0lstm_118/zeros_1:output:0!lstm_118/strided_slice_1:output:0@lstm_118/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_118_lstm_cell_123_matmul_readvariableop_resource7lstm_118_lstm_cell_123_matmul_1_readvariableop_resource6lstm_118_lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_118_while_body_23347008*(
cond R
lstm_118_while_cond_23347007*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations К
9lstm_118/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ё
+lstm_118/TensorArrayV2Stack/TensorListStackTensorListStacklstm_118/while:output:3Blstm_118/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0q
lstm_118/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 lstm_118/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_118/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
lstm_118/strided_slice_3StridedSlice4lstm_118/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_118/strided_slice_3/stack:output:0)lstm_118/strided_slice_3/stack_1:output:0)lstm_118/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskn
lstm_118/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ≤
lstm_118/transpose_1	Transpose4lstm_118/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_118/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іd
lstm_118/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_119/ShapeShapelstm_118/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_119/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_119/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_119/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
lstm_119/strided_sliceStridedSlicelstm_119/Shape:output:0%lstm_119/strided_slice/stack:output:0'lstm_119/strided_slice/stack_1:output:0'lstm_119/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
lstm_119/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іО
lstm_119/zeros/packedPacklstm_119/strided_slice:output:0 lstm_119/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_119/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
lstm_119/zerosFilllstm_119/zeros/packed:output:0lstm_119/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€і\
lstm_119/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іТ
lstm_119/zeros_1/packedPacklstm_119/strided_slice:output:0"lstm_119/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_119/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    О
lstm_119/zeros_1Fill lstm_119/zeros_1/packed:output:0lstm_119/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іl
lstm_119/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Т
lstm_119/transpose	Transposelstm_118/transpose_1:y:0 lstm_119/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іV
lstm_119/Shape_1Shapelstm_119/transpose:y:0*
T0*
_output_shapes
:h
lstm_119/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_119/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_119/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
lstm_119/strided_slice_1StridedSlicelstm_119/Shape_1:output:0'lstm_119/strided_slice_1/stack:output:0)lstm_119/strided_slice_1/stack_1:output:0)lstm_119/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_119/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
lstm_119/TensorArrayV2TensorListReserve-lstm_119/TensorArrayV2/element_shape:output:0!lstm_119/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“П
>lstm_119/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ы
0lstm_119/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_119/transpose:y:0Glstm_119/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“h
lstm_119/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_119/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_119/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
lstm_119/strided_slice_2StridedSlicelstm_119/transpose:y:0'lstm_119/strided_slice_2/stack:output:0)lstm_119/strided_slice_2/stack_1:output:0)lstm_119/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_mask§
,lstm_119/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp5lstm_119_lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≥
lstm_119/lstm_cell_124/MatMulMatMul!lstm_119/strided_slice_2:output:04lstm_119/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–®
.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp7lstm_119_lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≠
lstm_119/lstm_cell_124/MatMul_1MatMullstm_119/zeros:output:06lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–™
lstm_119/lstm_cell_124/addAddV2'lstm_119/lstm_cell_124/MatMul:product:0)lstm_119/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–°
-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp6lstm_119_lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0≥
lstm_119/lstm_cell_124/BiasAddBiasAddlstm_119/lstm_cell_124/add:z:05lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–h
&lstm_119/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :€
lstm_119/lstm_cell_124/splitSplit/lstm_119/lstm_cell_124/split/split_dim:output:0'lstm_119/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitГ
lstm_119/lstm_cell_124/SigmoidSigmoid%lstm_119/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_119/lstm_cell_124/Sigmoid_1Sigmoid%lstm_119/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іХ
lstm_119/lstm_cell_124/mulMul$lstm_119/lstm_cell_124/Sigmoid_1:y:0lstm_119/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€і}
lstm_119/lstm_cell_124/ReluRelu%lstm_119/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€і•
lstm_119/lstm_cell_124/mul_1Mul"lstm_119/lstm_cell_124/Sigmoid:y:0)lstm_119/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЪ
lstm_119/lstm_cell_124/add_1AddV2lstm_119/lstm_cell_124/mul:z:0 lstm_119/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_119/lstm_cell_124/Sigmoid_2Sigmoid%lstm_119/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_119/lstm_cell_124/Relu_1Relu lstm_119/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і©
lstm_119/lstm_cell_124/mul_2Mul$lstm_119/lstm_cell_124/Sigmoid_2:y:0+lstm_119/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іw
&lstm_119/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   g
%lstm_119/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :а
lstm_119/TensorArrayV2_1TensorListReserve/lstm_119/TensorArrayV2_1/element_shape:output:0.lstm_119/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“O
lstm_119/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_119/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
lstm_119/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Л
lstm_119/whileWhile$lstm_119/while/loop_counter:output:0*lstm_119/while/maximum_iterations:output:0lstm_119/time:output:0!lstm_119/TensorArrayV2_1:handle:0lstm_119/zeros:output:0lstm_119/zeros_1:output:0!lstm_119/strided_slice_1:output:0@lstm_119/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_119_lstm_cell_124_matmul_readvariableop_resource7lstm_119_lstm_cell_124_matmul_1_readvariableop_resource6lstm_119_lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_119_while_body_23347148*(
cond R
lstm_119_while_cond_23347147*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations К
9lstm_119/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   т
+lstm_119/TensorArrayV2Stack/TensorListStackTensorListStacklstm_119/while:output:3Blstm_119/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0*
num_elementsq
lstm_119/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 lstm_119/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_119/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
lstm_119/strided_slice_3StridedSlice4lstm_119/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_119/strided_slice_3/stack:output:0)lstm_119/strided_slice_3/stack_1:output:0)lstm_119/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskn
lstm_119/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ≤
lstm_119/transpose_1	Transpose4lstm_119/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_119/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іd
lstm_119/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    u
dropout_72/IdentityIdentity!lstm_119/strided_slice_3:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЗ
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes
:	і*
dtype0С
dense_89/MatMulMatMuldropout_72/Identity:output:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€м
NoOpNoOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp.^lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp-^lstm_117/lstm_cell_122/MatMul/ReadVariableOp/^lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp^lstm_117/while.^lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp-^lstm_118/lstm_cell_123/MatMul/ReadVariableOp/^lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp^lstm_118/while.^lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp-^lstm_119/lstm_cell_124/MatMul/ReadVariableOp/^lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp^lstm_119/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp2^
-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp2\
,lstm_117/lstm_cell_122/MatMul/ReadVariableOp,lstm_117/lstm_cell_122/MatMul/ReadVariableOp2`
.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp2 
lstm_117/whilelstm_117/while2^
-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp2\
,lstm_118/lstm_cell_123/MatMul/ReadVariableOp,lstm_118/lstm_cell_123/MatMul/ReadVariableOp2`
.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp2 
lstm_118/whilelstm_118/while2^
-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp2\
,lstm_119/lstm_cell_124/MatMul/ReadVariableOp,lstm_119/lstm_cell_124/MatMul/ReadVariableOp2`
.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp2 
lstm_119/whilelstm_119/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ƒK
®
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348623
inputs_0@
,lstm_cell_123_matmul_readvariableop_resource:
і–B
.lstm_cell_123_matmul_1_readvariableop_resource:
і–<
-lstm_cell_123_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_123/BiasAdd/ReadVariableOpҐ#lstm_cell_123/MatMul/ReadVariableOpҐ%lstm_cell_123/MatMul_1/ReadVariableOpҐwhile=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_123/MatMul/ReadVariableOpReadVariableOp,lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_123/MatMulMatMulstrided_slice_2:output:0+lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_123/MatMul_1MatMulzeros:output:0-lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_123/addAddV2lstm_cell_123/MatMul:product:0 lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_123/BiasAddBiasAddlstm_cell_123/add:z:0,lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_123/splitSplit&lstm_cell_123/split/split_dim:output:0lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_123/SigmoidSigmoidlstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_1Sigmoidlstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_123/mulMullstm_cell_123/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_123/ReluRelulstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_123/mul_1Mullstm_cell_123/Sigmoid:y:0 lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_123/add_1AddV2lstm_cell_123/mul:z:0lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_2Sigmoidlstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_123/Relu_1Relulstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_123/mul_2Mullstm_cell_123/Sigmoid_2:y:0"lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_123_matmul_readvariableop_resource.lstm_cell_123_matmul_1_readvariableop_resource-lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23348539*
condR
while_cond_23348538*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і√
NoOpNoOp%^lstm_cell_123/BiasAdd/ReadVariableOp$^lstm_cell_123/MatMul/ReadVariableOp&^lstm_cell_123/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2L
$lstm_cell_123/BiasAdd/ReadVariableOp$lstm_cell_123/BiasAdd/ReadVariableOp2J
#lstm_cell_123/MatMul/ReadVariableOp#lstm_cell_123/MatMul/ReadVariableOp2N
%lstm_cell_123/MatMul_1/ReadVariableOp%lstm_cell_123/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
"
_user_specified_name
inputs_0
ъ
Л
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349841

inputs
states_0
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
и8
Ё
while_body_23347923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_122_matmul_readvariableop_resource_0:	–J
6while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_122_matmul_readvariableop_resource:	–H
4while_lstm_cell_122_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_122/BiasAdd/ReadVariableOpҐ)while/lstm_cell_122/MatMul/ReadVariableOpҐ+while/lstm_cell_122/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Я
)while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0Љ
while/lstm_cell_122/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_122/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_122/addAddV2$while/lstm_cell_122/MatMul:product:0&while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_122/BiasAddBiasAddwhile/lstm_cell_122/add:z:02while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_122/splitSplit,while/lstm_cell_122/split/split_dim:output:0$while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_122/SigmoidSigmoid"while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_1Sigmoid"while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_122/mulMul!while/lstm_cell_122/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_122/ReluRelu"while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_122/mul_1Mulwhile/lstm_cell_122/Sigmoid:y:0&while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_122/add_1AddV2while/lstm_cell_122/mul:z:0while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_2Sigmoid"while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_122/Relu_1Reluwhile/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_122/mul_2Mul!while/lstm_cell_122/Sigmoid_2:y:0(while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_122/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_122/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_122/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_122/BiasAdd/ReadVariableOp*^while/lstm_cell_122/MatMul/ReadVariableOp,^while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_122_biasadd_readvariableop_resource5while_lstm_cell_122_biasadd_readvariableop_resource_0"n
4while_lstm_cell_122_matmul_1_readvariableop_resource6while_lstm_cell_122_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_122_matmul_readvariableop_resource4while_lstm_cell_122_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_122/BiasAdd/ReadVariableOp*while/lstm_cell_122/BiasAdd/ReadVariableOp2V
)while/lstm_cell_122/MatMul/ReadVariableOp)while/lstm_cell_122/MatMul/ReadVariableOp2Z
+while/lstm_cell_122/MatMul_1/ReadVariableOp+while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
√
Ќ
while_cond_23347779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23347779___redundant_placeholder06
2while_while_cond_23347779___redundant_placeholder16
2while_while_cond_23347779___redundant_placeholder26
2while_while_cond_23347779___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ј

Б
lstm_119_while_cond_23347147.
*lstm_119_while_lstm_119_while_loop_counter4
0lstm_119_while_lstm_119_while_maximum_iterations
lstm_119_while_placeholder 
lstm_119_while_placeholder_1 
lstm_119_while_placeholder_2 
lstm_119_while_placeholder_30
,lstm_119_while_less_lstm_119_strided_slice_1H
Dlstm_119_while_lstm_119_while_cond_23347147___redundant_placeholder0H
Dlstm_119_while_lstm_119_while_cond_23347147___redundant_placeholder1H
Dlstm_119_while_lstm_119_while_cond_23347147___redundant_placeholder2H
Dlstm_119_while_lstm_119_while_cond_23347147___redundant_placeholder3
lstm_119_while_identity
Ж
lstm_119/while/LessLesslstm_119_while_placeholder,lstm_119_while_less_lstm_119_strided_slice_1*
T0*
_output_shapes
: ]
lstm_119/while/IdentityIdentitylstm_119/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_119_while_identity lstm_119/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ЄC
э

lstm_117_while_body_23346869.
*lstm_117_while_lstm_117_while_loop_counter4
0lstm_117_while_lstm_117_while_maximum_iterations
lstm_117_while_placeholder 
lstm_117_while_placeholder_1 
lstm_117_while_placeholder_2 
lstm_117_while_placeholder_3-
)lstm_117_while_lstm_117_strided_slice_1_0i
elstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0P
=lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0:	–S
?lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–M
>lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
lstm_117_while_identity
lstm_117_while_identity_1
lstm_117_while_identity_2
lstm_117_while_identity_3
lstm_117_while_identity_4
lstm_117_while_identity_5+
'lstm_117_while_lstm_117_strided_slice_1g
clstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensorN
;lstm_117_while_lstm_cell_122_matmul_readvariableop_resource:	–Q
=lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource:
і–K
<lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpҐ2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOpҐ4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOpС
@lstm_117/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ”
2lstm_117/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0lstm_117_while_placeholderIlstm_117/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0±
2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp=lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0„
#lstm_117/while/lstm_cell_122/MatMulMatMul9lstm_117/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ґ
4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp?lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Њ
%lstm_117/while/lstm_cell_122/MatMul_1MatMullstm_117_while_placeholder_2<lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Љ
 lstm_117/while/lstm_cell_122/addAddV2-lstm_117/while/lstm_cell_122/MatMul:product:0/lstm_117/while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–ѓ
3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp>lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0≈
$lstm_117/while/lstm_cell_122/BiasAddBiasAdd$lstm_117/while/lstm_cell_122/add:z:0;lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–n
,lstm_117/while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :С
"lstm_117/while/lstm_cell_122/splitSplit5lstm_117/while/lstm_cell_122/split/split_dim:output:0-lstm_117/while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitП
$lstm_117/while/lstm_cell_122/SigmoidSigmoid+lstm_117/while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_117/while/lstm_cell_122/Sigmoid_1Sigmoid+lstm_117/while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€і§
 lstm_117/while/lstm_cell_122/mulMul*lstm_117/while/lstm_cell_122/Sigmoid_1:y:0lstm_117_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іЙ
!lstm_117/while/lstm_cell_122/ReluRelu+lstm_117/while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЈ
"lstm_117/while/lstm_cell_122/mul_1Mul(lstm_117/while/lstm_cell_122/Sigmoid:y:0/lstm_117/while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іђ
"lstm_117/while/lstm_cell_122/add_1AddV2$lstm_117/while/lstm_cell_122/mul:z:0&lstm_117/while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_117/while/lstm_cell_122/Sigmoid_2Sigmoid+lstm_117/while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЖ
#lstm_117/while/lstm_cell_122/Relu_1Relu&lstm_117/while/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ії
"lstm_117/while/lstm_cell_122/mul_2Mul*lstm_117/while/lstm_cell_122/Sigmoid_2:y:01lstm_117/while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ік
3lstm_117/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_117_while_placeholder_1lstm_117_while_placeholder&lstm_117/while/lstm_cell_122/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“V
lstm_117/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_117/while/addAddV2lstm_117_while_placeholderlstm_117/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_117/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
lstm_117/while/add_1AddV2*lstm_117_while_lstm_117_while_loop_counterlstm_117/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_117/while/IdentityIdentitylstm_117/while/add_1:z:0^lstm_117/while/NoOp*
T0*
_output_shapes
: О
lstm_117/while/Identity_1Identity0lstm_117_while_lstm_117_while_maximum_iterations^lstm_117/while/NoOp*
T0*
_output_shapes
: t
lstm_117/while/Identity_2Identitylstm_117/while/add:z:0^lstm_117/while/NoOp*
T0*
_output_shapes
: °
lstm_117/while/Identity_3IdentityClstm_117/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_117/while/NoOp*
T0*
_output_shapes
: Ц
lstm_117/while/Identity_4Identity&lstm_117/while/lstm_cell_122/mul_2:z:0^lstm_117/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іЦ
lstm_117/while/Identity_5Identity&lstm_117/while/lstm_cell_122/add_1:z:0^lstm_117/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іч
lstm_117/while/NoOpNoOp4^lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp3^lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp5^lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_117_while_identity lstm_117/while/Identity:output:0"?
lstm_117_while_identity_1"lstm_117/while/Identity_1:output:0"?
lstm_117_while_identity_2"lstm_117/while/Identity_2:output:0"?
lstm_117_while_identity_3"lstm_117/while/Identity_3:output:0"?
lstm_117_while_identity_4"lstm_117/while/Identity_4:output:0"?
lstm_117_while_identity_5"lstm_117/while/Identity_5:output:0"T
'lstm_117_while_lstm_117_strided_slice_1)lstm_117_while_lstm_117_strided_slice_1_0"~
<lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource>lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0"А
=lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource?lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0"|
;lstm_117_while_lstm_cell_122_matmul_readvariableop_resource=lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0"ћ
clstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensorelstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2j
3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp2h
2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp2l
4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
м8
я
while_body_23348539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_123_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_123_matmul_readvariableop_resource:
і–H
4while_lstm_cell_123_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_123/BiasAdd/ReadVariableOpҐ)while/lstm_cell_123/MatMul/ReadVariableOpҐ+while/lstm_cell_123/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_123/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_123/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_123/addAddV2$while/lstm_cell_123/MatMul:product:0&while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_123/BiasAddBiasAddwhile/lstm_cell_123/add:z:02while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_123/splitSplit,while/lstm_cell_123/split/split_dim:output:0$while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_123/SigmoidSigmoid"while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_1Sigmoid"while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_123/mulMul!while/lstm_cell_123/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_123/ReluRelu"while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_123/mul_1Mulwhile/lstm_cell_123/Sigmoid:y:0&while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_123/add_1AddV2while/lstm_cell_123/mul:z:0while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_2Sigmoid"while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_123/Relu_1Reluwhile/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_123/mul_2Mul!while/lstm_cell_123/Sigmoid_2:y:0(while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_123/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_123/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_123/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_123/BiasAdd/ReadVariableOp*^while/lstm_cell_123/MatMul/ReadVariableOp,^while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_123_biasadd_readvariableop_resource5while_lstm_cell_123_biasadd_readvariableop_resource_0"n
4while_lstm_cell_123_matmul_1_readvariableop_resource6while_lstm_cell_123_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_123_matmul_readvariableop_resource4while_lstm_cell_123_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_123/BiasAdd/ReadVariableOp*while/lstm_cell_123/BiasAdd/ReadVariableOp2V
)while/lstm_cell_123/MatMul/ReadVariableOp)while/lstm_cell_123/MatMul/ReadVariableOp2Z
+while/lstm_cell_123/MatMul_1/ReadVariableOp+while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ЄC
э

lstm_117_while_body_23347299.
*lstm_117_while_lstm_117_while_loop_counter4
0lstm_117_while_lstm_117_while_maximum_iterations
lstm_117_while_placeholder 
lstm_117_while_placeholder_1 
lstm_117_while_placeholder_2 
lstm_117_while_placeholder_3-
)lstm_117_while_lstm_117_strided_slice_1_0i
elstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0P
=lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0:	–S
?lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–M
>lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
lstm_117_while_identity
lstm_117_while_identity_1
lstm_117_while_identity_2
lstm_117_while_identity_3
lstm_117_while_identity_4
lstm_117_while_identity_5+
'lstm_117_while_lstm_117_strided_slice_1g
clstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensorN
;lstm_117_while_lstm_cell_122_matmul_readvariableop_resource:	–Q
=lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource:
і–K
<lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpҐ2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOpҐ4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOpС
@lstm_117/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ”
2lstm_117/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0lstm_117_while_placeholderIlstm_117/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0±
2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp=lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0„
#lstm_117/while/lstm_cell_122/MatMulMatMul9lstm_117/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ґ
4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp?lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Њ
%lstm_117/while/lstm_cell_122/MatMul_1MatMullstm_117_while_placeholder_2<lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Љ
 lstm_117/while/lstm_cell_122/addAddV2-lstm_117/while/lstm_cell_122/MatMul:product:0/lstm_117/while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–ѓ
3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp>lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0≈
$lstm_117/while/lstm_cell_122/BiasAddBiasAdd$lstm_117/while/lstm_cell_122/add:z:0;lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–n
,lstm_117/while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :С
"lstm_117/while/lstm_cell_122/splitSplit5lstm_117/while/lstm_cell_122/split/split_dim:output:0-lstm_117/while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitП
$lstm_117/while/lstm_cell_122/SigmoidSigmoid+lstm_117/while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_117/while/lstm_cell_122/Sigmoid_1Sigmoid+lstm_117/while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€і§
 lstm_117/while/lstm_cell_122/mulMul*lstm_117/while/lstm_cell_122/Sigmoid_1:y:0lstm_117_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іЙ
!lstm_117/while/lstm_cell_122/ReluRelu+lstm_117/while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЈ
"lstm_117/while/lstm_cell_122/mul_1Mul(lstm_117/while/lstm_cell_122/Sigmoid:y:0/lstm_117/while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іђ
"lstm_117/while/lstm_cell_122/add_1AddV2$lstm_117/while/lstm_cell_122/mul:z:0&lstm_117/while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_117/while/lstm_cell_122/Sigmoid_2Sigmoid+lstm_117/while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЖ
#lstm_117/while/lstm_cell_122/Relu_1Relu&lstm_117/while/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ії
"lstm_117/while/lstm_cell_122/mul_2Mul*lstm_117/while/lstm_cell_122/Sigmoid_2:y:01lstm_117/while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ік
3lstm_117/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_117_while_placeholder_1lstm_117_while_placeholder&lstm_117/while/lstm_cell_122/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“V
lstm_117/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_117/while/addAddV2lstm_117_while_placeholderlstm_117/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_117/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
lstm_117/while/add_1AddV2*lstm_117_while_lstm_117_while_loop_counterlstm_117/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_117/while/IdentityIdentitylstm_117/while/add_1:z:0^lstm_117/while/NoOp*
T0*
_output_shapes
: О
lstm_117/while/Identity_1Identity0lstm_117_while_lstm_117_while_maximum_iterations^lstm_117/while/NoOp*
T0*
_output_shapes
: t
lstm_117/while/Identity_2Identitylstm_117/while/add:z:0^lstm_117/while/NoOp*
T0*
_output_shapes
: °
lstm_117/while/Identity_3IdentityClstm_117/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_117/while/NoOp*
T0*
_output_shapes
: Ц
lstm_117/while/Identity_4Identity&lstm_117/while/lstm_cell_122/mul_2:z:0^lstm_117/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іЦ
lstm_117/while/Identity_5Identity&lstm_117/while/lstm_cell_122/add_1:z:0^lstm_117/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іч
lstm_117/while/NoOpNoOp4^lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp3^lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp5^lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_117_while_identity lstm_117/while/Identity:output:0"?
lstm_117_while_identity_1"lstm_117/while/Identity_1:output:0"?
lstm_117_while_identity_2"lstm_117/while/Identity_2:output:0"?
lstm_117_while_identity_3"lstm_117/while/Identity_3:output:0"?
lstm_117_while_identity_4"lstm_117/while/Identity_4:output:0"?
lstm_117_while_identity_5"lstm_117/while/Identity_5:output:0"T
'lstm_117_while_lstm_117_strided_slice_1)lstm_117_while_lstm_117_strided_slice_1_0"~
<lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource>lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0"А
=lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource?lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0"|
;lstm_117_while_lstm_cell_122_matmul_readvariableop_resource=lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0"ћ
clstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensorelstm_117_while_tensorarrayv2read_tensorlistgetitem_lstm_117_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2j
3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp3lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp2h
2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp2lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp2l
4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp4lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
о
И
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344672

inputs

states
states_11
matmul_readvariableop_resource:	–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates
√
Ќ
while_cond_23348824
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23348824___redundant_placeholder06
2while_while_cond_23348824___redundant_placeholder16
2while_while_cond_23348824___redundant_placeholder26
2while_while_cond_23348824___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
©
I
-__inference_dropout_72_layer_call_fn_23349538

inputs
identityі
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_72_layer_call_and_return_conditional_losses_23345977a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€і"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€і:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
я™
е
$__inference__traced_restore_23350140
file_prefix3
 assignvariableop_dense_89_kernel:	і.
 assignvariableop_1_dense_89_bias:C
0assignvariableop_2_lstm_117_lstm_cell_122_kernel:	–N
:assignvariableop_3_lstm_117_lstm_cell_122_recurrent_kernel:
і–=
.assignvariableop_4_lstm_117_lstm_cell_122_bias:	–D
0assignvariableop_5_lstm_118_lstm_cell_123_kernel:
і–N
:assignvariableop_6_lstm_118_lstm_cell_123_recurrent_kernel:
і–=
.assignvariableop_7_lstm_118_lstm_cell_123_bias:	–D
0assignvariableop_8_lstm_119_lstm_cell_124_kernel:
і–N
:assignvariableop_9_lstm_119_lstm_cell_124_recurrent_kernel:
і–>
/assignvariableop_10_lstm_119_lstm_cell_124_bias:	–'
assignvariableop_11_iteration:	 +
!assignvariableop_12_learning_rate: K
8assignvariableop_13_adam_m_lstm_117_lstm_cell_122_kernel:	–K
8assignvariableop_14_adam_v_lstm_117_lstm_cell_122_kernel:	–V
Bassignvariableop_15_adam_m_lstm_117_lstm_cell_122_recurrent_kernel:
і–V
Bassignvariableop_16_adam_v_lstm_117_lstm_cell_122_recurrent_kernel:
і–E
6assignvariableop_17_adam_m_lstm_117_lstm_cell_122_bias:	–E
6assignvariableop_18_adam_v_lstm_117_lstm_cell_122_bias:	–L
8assignvariableop_19_adam_m_lstm_118_lstm_cell_123_kernel:
і–L
8assignvariableop_20_adam_v_lstm_118_lstm_cell_123_kernel:
і–V
Bassignvariableop_21_adam_m_lstm_118_lstm_cell_123_recurrent_kernel:
і–V
Bassignvariableop_22_adam_v_lstm_118_lstm_cell_123_recurrent_kernel:
і–E
6assignvariableop_23_adam_m_lstm_118_lstm_cell_123_bias:	–E
6assignvariableop_24_adam_v_lstm_118_lstm_cell_123_bias:	–L
8assignvariableop_25_adam_m_lstm_119_lstm_cell_124_kernel:
і–L
8assignvariableop_26_adam_v_lstm_119_lstm_cell_124_kernel:
і–V
Bassignvariableop_27_adam_m_lstm_119_lstm_cell_124_recurrent_kernel:
і–V
Bassignvariableop_28_adam_v_lstm_119_lstm_cell_124_recurrent_kernel:
і–E
6assignvariableop_29_adam_m_lstm_119_lstm_cell_124_bias:	–E
6assignvariableop_30_adam_v_lstm_119_lstm_cell_124_bias:	–=
*assignvariableop_31_adam_m_dense_89_kernel:	і=
*assignvariableop_32_adam_v_dense_89_kernel:	і6
(assignvariableop_33_adam_m_dense_89_bias:6
(assignvariableop_34_adam_v_dense_89_bias:%
assignvariableop_35_total_1: %
assignvariableop_36_count_1: #
assignvariableop_37_total: #
assignvariableop_38_count: 
identity_40ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9С
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Ј
value≠B™(B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHј
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B й
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:≥
AssignVariableOpAssignVariableOp assignvariableop_dense_89_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_89_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_2AssignVariableOp0assignvariableop_2_lstm_117_lstm_cell_122_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_3AssignVariableOp:assignvariableop_3_lstm_117_lstm_cell_122_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_4AssignVariableOp.assignvariableop_4_lstm_117_lstm_cell_122_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_5AssignVariableOp0assignvariableop_5_lstm_118_lstm_cell_123_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_6AssignVariableOp:assignvariableop_6_lstm_118_lstm_cell_123_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_118_lstm_cell_123_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_8AssignVariableOp0assignvariableop_8_lstm_119_lstm_cell_124_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_9AssignVariableOp:assignvariableop_9_lstm_119_lstm_cell_124_recurrent_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_119_lstm_cell_124_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_11AssignVariableOpassignvariableop_11_iterationIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_12AssignVariableOp!assignvariableop_12_learning_rateIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_m_lstm_117_lstm_cell_122_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_14AssignVariableOp8assignvariableop_14_adam_v_lstm_117_lstm_cell_122_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_15AssignVariableOpBassignvariableop_15_adam_m_lstm_117_lstm_cell_122_recurrent_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_16AssignVariableOpBassignvariableop_16_adam_v_lstm_117_lstm_cell_122_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_m_lstm_117_lstm_cell_122_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_v_lstm_117_lstm_cell_122_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_m_lstm_118_lstm_cell_123_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_v_lstm_118_lstm_cell_123_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_21AssignVariableOpBassignvariableop_21_adam_m_lstm_118_lstm_cell_123_recurrent_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_22AssignVariableOpBassignvariableop_22_adam_v_lstm_118_lstm_cell_123_recurrent_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_m_lstm_118_lstm_cell_123_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_v_lstm_118_lstm_cell_123_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_m_lstm_119_lstm_cell_124_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:—
AssignVariableOp_26AssignVariableOp8assignvariableop_26_adam_v_lstm_119_lstm_cell_124_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_27AssignVariableOpBassignvariableop_27_adam_m_lstm_119_lstm_cell_124_recurrent_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_v_lstm_119_lstm_cell_124_recurrent_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_m_lstm_119_lstm_cell_124_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_v_lstm_119_lstm_cell_124_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_m_dense_89_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_v_dense_89_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_m_dense_89_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_v_dense_89_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ©
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: Ц
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
ЉC
€

lstm_118_while_body_23347438.
*lstm_118_while_lstm_118_while_loop_counter4
0lstm_118_while_lstm_118_while_maximum_iterations
lstm_118_while_placeholder 
lstm_118_while_placeholder_1 
lstm_118_while_placeholder_2 
lstm_118_while_placeholder_3-
)lstm_118_while_lstm_118_strided_slice_1_0i
elstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0:
і–S
?lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–M
>lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
lstm_118_while_identity
lstm_118_while_identity_1
lstm_118_while_identity_2
lstm_118_while_identity_3
lstm_118_while_identity_4
lstm_118_while_identity_5+
'lstm_118_while_lstm_118_strided_slice_1g
clstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensorO
;lstm_118_while_lstm_cell_123_matmul_readvariableop_resource:
і–Q
=lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource:
і–K
<lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpҐ2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOpҐ4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOpС
@lstm_118/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ‘
2lstm_118/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensor_0lstm_118_while_placeholderIlstm_118/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0≤
2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp=lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0„
#lstm_118/while/lstm_cell_123/MatMulMatMul9lstm_118/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ґ
4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp?lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Њ
%lstm_118/while/lstm_cell_123/MatMul_1MatMullstm_118_while_placeholder_2<lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Љ
 lstm_118/while/lstm_cell_123/addAddV2-lstm_118/while/lstm_cell_123/MatMul:product:0/lstm_118/while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–ѓ
3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp>lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0≈
$lstm_118/while/lstm_cell_123/BiasAddBiasAdd$lstm_118/while/lstm_cell_123/add:z:0;lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–n
,lstm_118/while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :С
"lstm_118/while/lstm_cell_123/splitSplit5lstm_118/while/lstm_cell_123/split/split_dim:output:0-lstm_118/while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitП
$lstm_118/while/lstm_cell_123/SigmoidSigmoid+lstm_118/while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_118/while/lstm_cell_123/Sigmoid_1Sigmoid+lstm_118/while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€і§
 lstm_118/while/lstm_cell_123/mulMul*lstm_118/while/lstm_cell_123/Sigmoid_1:y:0lstm_118_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іЙ
!lstm_118/while/lstm_cell_123/ReluRelu+lstm_118/while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЈ
"lstm_118/while/lstm_cell_123/mul_1Mul(lstm_118/while/lstm_cell_123/Sigmoid:y:0/lstm_118/while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іђ
"lstm_118/while/lstm_cell_123/add_1AddV2$lstm_118/while/lstm_cell_123/mul:z:0&lstm_118/while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_118/while/lstm_cell_123/Sigmoid_2Sigmoid+lstm_118/while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЖ
#lstm_118/while/lstm_cell_123/Relu_1Relu&lstm_118/while/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ії
"lstm_118/while/lstm_cell_123/mul_2Mul*lstm_118/while/lstm_cell_123/Sigmoid_2:y:01lstm_118/while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ік
3lstm_118/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_118_while_placeholder_1lstm_118_while_placeholder&lstm_118/while/lstm_cell_123/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“V
lstm_118/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_118/while/addAddV2lstm_118_while_placeholderlstm_118/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_118/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
lstm_118/while/add_1AddV2*lstm_118_while_lstm_118_while_loop_counterlstm_118/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_118/while/IdentityIdentitylstm_118/while/add_1:z:0^lstm_118/while/NoOp*
T0*
_output_shapes
: О
lstm_118/while/Identity_1Identity0lstm_118_while_lstm_118_while_maximum_iterations^lstm_118/while/NoOp*
T0*
_output_shapes
: t
lstm_118/while/Identity_2Identitylstm_118/while/add:z:0^lstm_118/while/NoOp*
T0*
_output_shapes
: °
lstm_118/while/Identity_3IdentityClstm_118/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_118/while/NoOp*
T0*
_output_shapes
: Ц
lstm_118/while/Identity_4Identity&lstm_118/while/lstm_cell_123/mul_2:z:0^lstm_118/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іЦ
lstm_118/while/Identity_5Identity&lstm_118/while/lstm_cell_123/add_1:z:0^lstm_118/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іч
lstm_118/while/NoOpNoOp4^lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp3^lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp5^lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_118_while_identity lstm_118/while/Identity:output:0"?
lstm_118_while_identity_1"lstm_118/while/Identity_1:output:0"?
lstm_118_while_identity_2"lstm_118/while/Identity_2:output:0"?
lstm_118_while_identity_3"lstm_118/while/Identity_3:output:0"?
lstm_118_while_identity_4"lstm_118/while/Identity_4:output:0"?
lstm_118_while_identity_5"lstm_118/while/Identity_5:output:0"T
'lstm_118_while_lstm_118_strided_slice_1)lstm_118_while_lstm_118_strided_slice_1_0"~
<lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource>lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0"А
=lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource?lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0"|
;lstm_118_while_lstm_cell_123_matmul_readvariableop_resource=lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0"ћ
clstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensorelstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2j
3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp2h
2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp2l
4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ц
К
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349645

inputs
states_0
states_11
matmul_readvariableop_resource:	–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
ъS
ј
*sequential_91_lstm_117_while_body_23344088J
Fsequential_91_lstm_117_while_sequential_91_lstm_117_while_loop_counterP
Lsequential_91_lstm_117_while_sequential_91_lstm_117_while_maximum_iterations,
(sequential_91_lstm_117_while_placeholder.
*sequential_91_lstm_117_while_placeholder_1.
*sequential_91_lstm_117_while_placeholder_2.
*sequential_91_lstm_117_while_placeholder_3I
Esequential_91_lstm_117_while_sequential_91_lstm_117_strided_slice_1_0Ж
Бsequential_91_lstm_117_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_117_tensorarrayunstack_tensorlistfromtensor_0^
Ksequential_91_lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0:	–a
Msequential_91_lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–[
Lsequential_91_lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0:	–)
%sequential_91_lstm_117_while_identity+
'sequential_91_lstm_117_while_identity_1+
'sequential_91_lstm_117_while_identity_2+
'sequential_91_lstm_117_while_identity_3+
'sequential_91_lstm_117_while_identity_4+
'sequential_91_lstm_117_while_identity_5G
Csequential_91_lstm_117_while_sequential_91_lstm_117_strided_slice_1Г
sequential_91_lstm_117_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_117_tensorarrayunstack_tensorlistfromtensor\
Isequential_91_lstm_117_while_lstm_cell_122_matmul_readvariableop_resource:	–_
Ksequential_91_lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource:
і–Y
Jsequential_91_lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐAsequential_91/lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpҐ@sequential_91/lstm_117/while/lstm_cell_122/MatMul/ReadVariableOpҐBsequential_91/lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOpЯ
Nsequential_91/lstm_117/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Ъ
@sequential_91/lstm_117/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemБsequential_91_lstm_117_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_117_tensorarrayunstack_tensorlistfromtensor_0(sequential_91_lstm_117_while_placeholderWsequential_91/lstm_117/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Ќ
@sequential_91/lstm_117/while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOpKsequential_91_lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0Б
1sequential_91/lstm_117/while/lstm_cell_122/MatMulMatMulGsequential_91/lstm_117/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_91/lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–“
Bsequential_91/lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOpMsequential_91_lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0и
3sequential_91/lstm_117/while/lstm_cell_122/MatMul_1MatMul*sequential_91_lstm_117_while_placeholder_2Jsequential_91/lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ж
.sequential_91/lstm_117/while/lstm_cell_122/addAddV2;sequential_91/lstm_117/while/lstm_cell_122/MatMul:product:0=sequential_91/lstm_117/while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Ћ
Asequential_91/lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOpLsequential_91_lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0п
2sequential_91/lstm_117/while/lstm_cell_122/BiasAddBiasAdd2sequential_91/lstm_117/while/lstm_cell_122/add:z:0Isequential_91/lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–|
:sequential_91/lstm_117/while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
0sequential_91/lstm_117/while/lstm_cell_122/splitSplitCsequential_91/lstm_117/while/lstm_cell_122/split/split_dim:output:0;sequential_91/lstm_117/while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitЂ
2sequential_91/lstm_117/while/lstm_cell_122/SigmoidSigmoid9sequential_91/lstm_117/while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і≠
4sequential_91/lstm_117/while/lstm_cell_122/Sigmoid_1Sigmoid9sequential_91/lstm_117/while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іќ
.sequential_91/lstm_117/while/lstm_cell_122/mulMul8sequential_91/lstm_117/while/lstm_cell_122/Sigmoid_1:y:0*sequential_91_lstm_117_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€і•
/sequential_91/lstm_117/while/lstm_cell_122/ReluRelu9sequential_91/lstm_117/while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іб
0sequential_91/lstm_117/while/lstm_cell_122/mul_1Mul6sequential_91/lstm_117/while/lstm_cell_122/Sigmoid:y:0=sequential_91/lstm_117/while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і÷
0sequential_91/lstm_117/while/lstm_cell_122/add_1AddV22sequential_91/lstm_117/while/lstm_cell_122/mul:z:04sequential_91/lstm_117/while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і≠
4sequential_91/lstm_117/while/lstm_cell_122/Sigmoid_2Sigmoid9sequential_91/lstm_117/while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іҐ
1sequential_91/lstm_117/while/lstm_cell_122/Relu_1Relu4sequential_91/lstm_117/while/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іе
0sequential_91/lstm_117/while/lstm_cell_122/mul_2Mul8sequential_91/lstm_117/while/lstm_cell_122/Sigmoid_2:y:0?sequential_91/lstm_117/while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іҐ
Asequential_91/lstm_117/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_91_lstm_117_while_placeholder_1(sequential_91_lstm_117_while_placeholder4sequential_91/lstm_117/while/lstm_cell_122/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“d
"sequential_91/lstm_117/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :°
 sequential_91/lstm_117/while/addAddV2(sequential_91_lstm_117_while_placeholder+sequential_91/lstm_117/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_91/lstm_117/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :√
"sequential_91/lstm_117/while/add_1AddV2Fsequential_91_lstm_117_while_sequential_91_lstm_117_while_loop_counter-sequential_91/lstm_117/while/add_1/y:output:0*
T0*
_output_shapes
: Ю
%sequential_91/lstm_117/while/IdentityIdentity&sequential_91/lstm_117/while/add_1:z:0"^sequential_91/lstm_117/while/NoOp*
T0*
_output_shapes
: ∆
'sequential_91/lstm_117/while/Identity_1IdentityLsequential_91_lstm_117_while_sequential_91_lstm_117_while_maximum_iterations"^sequential_91/lstm_117/while/NoOp*
T0*
_output_shapes
: Ю
'sequential_91/lstm_117/while/Identity_2Identity$sequential_91/lstm_117/while/add:z:0"^sequential_91/lstm_117/while/NoOp*
T0*
_output_shapes
: Ћ
'sequential_91/lstm_117/while/Identity_3IdentityQsequential_91/lstm_117/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_91/lstm_117/while/NoOp*
T0*
_output_shapes
: ј
'sequential_91/lstm_117/while/Identity_4Identity4sequential_91/lstm_117/while/lstm_cell_122/mul_2:z:0"^sequential_91/lstm_117/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іј
'sequential_91/lstm_117/while/Identity_5Identity4sequential_91/lstm_117/while/lstm_cell_122/add_1:z:0"^sequential_91/lstm_117/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іѓ
!sequential_91/lstm_117/while/NoOpNoOpB^sequential_91/lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpA^sequential_91/lstm_117/while/lstm_cell_122/MatMul/ReadVariableOpC^sequential_91/lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_91_lstm_117_while_identity.sequential_91/lstm_117/while/Identity:output:0"[
'sequential_91_lstm_117_while_identity_10sequential_91/lstm_117/while/Identity_1:output:0"[
'sequential_91_lstm_117_while_identity_20sequential_91/lstm_117/while/Identity_2:output:0"[
'sequential_91_lstm_117_while_identity_30sequential_91/lstm_117/while/Identity_3:output:0"[
'sequential_91_lstm_117_while_identity_40sequential_91/lstm_117/while/Identity_4:output:0"[
'sequential_91_lstm_117_while_identity_50sequential_91/lstm_117/while/Identity_5:output:0"Ъ
Jsequential_91_lstm_117_while_lstm_cell_122_biasadd_readvariableop_resourceLsequential_91_lstm_117_while_lstm_cell_122_biasadd_readvariableop_resource_0"Ь
Ksequential_91_lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resourceMsequential_91_lstm_117_while_lstm_cell_122_matmul_1_readvariableop_resource_0"Ш
Isequential_91_lstm_117_while_lstm_cell_122_matmul_readvariableop_resourceKsequential_91_lstm_117_while_lstm_cell_122_matmul_readvariableop_resource_0"М
Csequential_91_lstm_117_while_sequential_91_lstm_117_strided_slice_1Esequential_91_lstm_117_while_sequential_91_lstm_117_strided_slice_1_0"Е
sequential_91_lstm_117_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_117_tensorarrayunstack_tensorlistfromtensorБsequential_91_lstm_117_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_117_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Ж
Asequential_91/lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOpAsequential_91/lstm_117/while/lstm_cell_122/BiasAdd/ReadVariableOp2Д
@sequential_91/lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp@sequential_91/lstm_117/while/lstm_cell_122/MatMul/ReadVariableOp2И
Bsequential_91/lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOpBsequential_91/lstm_117/while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
Ѓ#
ь
while_body_23344890
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_123_23344914_0:
і–2
while_lstm_cell_123_23344916_0:
і–-
while_lstm_cell_123_23344918_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_123_23344914:
і–0
while_lstm_cell_123_23344916:
і–+
while_lstm_cell_123_23344918:	–ИҐ+while/lstm_cell_123/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0√
+while/lstm_cell_123/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_123_23344914_0while_lstm_cell_123_23344916_0while_lstm_cell_123_23344918_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23344876Ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder4while/lstm_cell_123/StatefulPartitionedCall:output:0*
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
: Т
while/Identity_4Identity4while/lstm_cell_123/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іТ
while/Identity_5Identity4while/lstm_cell_123/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іz

while/NoOpNoOp,^while/lstm_cell_123/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_123_23344914while_lstm_cell_123_23344914_0">
while_lstm_cell_123_23344916while_lstm_cell_123_23344916_0">
while_lstm_cell_123_23344918while_lstm_cell_123_23344918_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Z
+while/lstm_cell_123/StatefulPartitionedCall+while/lstm_cell_123/StatefulPartitionedCall: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
АK
•
F__inference_lstm_117_layer_call_and_return_conditional_losses_23346542

inputs?
,lstm_cell_122_matmul_readvariableop_resource:	–B
.lstm_cell_122_matmul_1_readvariableop_resource:
і–<
-lstm_cell_122_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_122/BiasAdd/ReadVariableOpҐ#lstm_cell_122/MatMul/ReadVariableOpҐ%lstm_cell_122/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskС
#lstm_cell_122/MatMul/ReadVariableOpReadVariableOp,lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0Ш
lstm_cell_122/MatMulMatMulstrided_slice_2:output:0+lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_122/MatMul_1MatMulzeros:output:0-lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_122/addAddV2lstm_cell_122/MatMul:product:0 lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_122/BiasAddBiasAddlstm_cell_122/add:z:0,lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_122/splitSplit&lstm_cell_122/split/split_dim:output:0lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_122/SigmoidSigmoidlstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_1Sigmoidlstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_122/mulMullstm_cell_122/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_122/ReluRelulstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_122/mul_1Mullstm_cell_122/Sigmoid:y:0 lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_122/add_1AddV2lstm_cell_122/mul:z:0lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_2Sigmoidlstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_122/Relu_1Relulstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_122/mul_2Mullstm_cell_122/Sigmoid_2:y:0"lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_122_matmul_readvariableop_resource.lstm_cell_122_matmul_1_readvariableop_resource-lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23346458*
condR
while_cond_23346457*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_122/BiasAdd/ReadVariableOp$^lstm_cell_122/MatMul/ReadVariableOp&^lstm_cell_122/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2L
$lstm_cell_122/BiasAdd/ReadVariableOp$lstm_cell_122/BiasAdd/ReadVariableOp2J
#lstm_cell_122/MatMul/ReadVariableOp#lstm_cell_122/MatMul/ReadVariableOp2N
%lstm_cell_122/MatMul_1/ReadVariableOp%lstm_cell_122/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ј

Б
lstm_117_while_cond_23346868.
*lstm_117_while_lstm_117_while_loop_counter4
0lstm_117_while_lstm_117_while_maximum_iterations
lstm_117_while_placeholder 
lstm_117_while_placeholder_1 
lstm_117_while_placeholder_2 
lstm_117_while_placeholder_30
,lstm_117_while_less_lstm_117_strided_slice_1H
Dlstm_117_while_lstm_117_while_cond_23346868___redundant_placeholder0H
Dlstm_117_while_lstm_117_while_cond_23346868___redundant_placeholder1H
Dlstm_117_while_lstm_117_while_cond_23346868___redundant_placeholder2H
Dlstm_117_while_lstm_117_while_cond_23346868___redundant_placeholder3
lstm_117_while_identity
Ж
lstm_117/while/LessLesslstm_117_while_placeholder,lstm_117_while_less_lstm_117_strided_slice_1*
T0*
_output_shapes
: ]
lstm_117/while/IdentityIdentitylstm_117/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_117_while_identity lstm_117/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ь
ъ
0__inference_lstm_cell_122_layer_call_fn_23349613

inputs
states_0
states_1
unknown:	–
	unknown_0:
і–
	unknown_1:	–
identity

identity_1

identity_2ИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344672p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€:€€€€€€€€€і:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
ъ
Л
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349873

inputs
states_0
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
ЖK
¶
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348909

inputs@
,lstm_cell_123_matmul_readvariableop_resource:
і–B
.lstm_cell_123_matmul_1_readvariableop_resource:
і–<
-lstm_cell_123_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_123/BiasAdd/ReadVariableOpҐ#lstm_cell_123/MatMul/ReadVariableOpҐ%lstm_cell_123/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_123/MatMul/ReadVariableOpReadVariableOp,lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_123/MatMulMatMulstrided_slice_2:output:0+lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_123/MatMul_1MatMulzeros:output:0-lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_123/addAddV2lstm_cell_123/MatMul:product:0 lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_123/BiasAddBiasAddlstm_cell_123/add:z:0,lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_123/splitSplit&lstm_cell_123/split/split_dim:output:0lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_123/SigmoidSigmoidlstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_1Sigmoidlstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_123/mulMullstm_cell_123/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_123/ReluRelulstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_123/mul_1Mullstm_cell_123/Sigmoid:y:0 lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_123/add_1AddV2lstm_cell_123/mul:z:0lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_123/Sigmoid_2Sigmoidlstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_123/Relu_1Relulstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_123/mul_2Mullstm_cell_123/Sigmoid_2:y:0"lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_123_matmul_readvariableop_resource.lstm_cell_123_matmul_1_readvariableop_resource-lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23348825*
condR
while_cond_23348824*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_123/BiasAdd/ReadVariableOp$^lstm_cell_123/MatMul/ReadVariableOp&^lstm_cell_123/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_123/BiasAdd/ReadVariableOp$lstm_cell_123/BiasAdd/ReadVariableOp2J
#lstm_cell_123/MatMul/ReadVariableOp#lstm_cell_123/MatMul/ReadVariableOp2N
%lstm_cell_123/MatMul_1/ReadVariableOp%lstm_cell_123/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
з

•
0__inference_sequential_91_layer_call_fn_23346783

inputs
unknown:	–
	unknown_0:
і–
	unknown_1:	–
	unknown_2:
і–
	unknown_3:
і–
	unknown_4:	–
	unknown_5:
і–
	unknown_6:
і–
	unknown_7:	–
	unknown_8:	і
	unknown_9:
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_91_layer_call_and_return_conditional_losses_23345996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ь
ъ
0__inference_lstm_cell_122_layer_call_fn_23349596

inputs
states_0
states_1
unknown:	–
	unknown_0:
і–
	unknown_1:	–
identity

identity_1

identity_2ИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344526p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€:€€€€€€€€€і:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_0:RN
(
_output_shapes
:€€€€€€€€€і
"
_user_specified_name
states_1
я
f
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349548

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€і\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€і"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€і:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
я
f
H__inference_dropout_72_layer_call_and_return_conditional_losses_23345977

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€і\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€і"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€і:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
√
Ќ
while_cond_23348395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23348395___redundant_placeholder06
2while_while_cond_23348395___redundant_placeholder16
2while_while_cond_23348395___redundant_placeholder26
2while_while_cond_23348395___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ш
±
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346611

inputs$
lstm_117_23346583:	–%
lstm_117_23346585:
і– 
lstm_117_23346587:	–%
lstm_118_23346590:
і–%
lstm_118_23346592:
і– 
lstm_118_23346594:	–%
lstm_119_23346597:
і–%
lstm_119_23346599:
і– 
lstm_119_23346601:	–$
dense_89_23346605:	і
dense_89_23346607:
identityИҐ dense_89/StatefulPartitionedCallҐ"dropout_72/StatefulPartitionedCallҐ lstm_117/StatefulPartitionedCallҐ lstm_118/StatefulPartitionedCallҐ lstm_119/StatefulPartitionedCallР
 lstm_117/StatefulPartitionedCallStatefulPartitionedCallinputslstm_117_23346583lstm_117_23346585lstm_117_23346587*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23346542≥
 lstm_118/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0lstm_118_23346590lstm_118_23346592lstm_118_23346594*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23346377ѓ
 lstm_119/StatefulPartitionedCallStatefulPartitionedCall)lstm_118/StatefulPartitionedCall:output:0lstm_119_23346597lstm_119_23346599lstm_119_23346601*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23346212т
"dropout_72/StatefulPartitionedCallStatefulPartitionedCall)lstm_119/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_72_layer_call_and_return_conditional_losses_23346051Ы
 dense_89/StatefulPartitionedCallStatefulPartitionedCall+dropout_72/StatefulPartitionedCall:output:0dense_89_23346605dense_89_23346607*
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
F__inference_dense_89_layer_call_and_return_conditional_losses_23345989x
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ч
NoOpNoOp!^dense_89/StatefulPartitionedCall#^dropout_72/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall!^lstm_118/StatefulPartitionedCall!^lstm_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2H
"dropout_72/StatefulPartitionedCall"dropout_72/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall2D
 lstm_118/StatefulPartitionedCall lstm_118/StatefulPartitionedCall2D
 lstm_119/StatefulPartitionedCall lstm_119/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м8
я
while_body_23348825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_123_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_123_matmul_readvariableop_resource:
і–H
4while_lstm_cell_123_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_123/BiasAdd/ReadVariableOpҐ)while/lstm_cell_123/MatMul/ReadVariableOpҐ+while/lstm_cell_123/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_123/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_123/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_123/addAddV2$while/lstm_cell_123/MatMul:product:0&while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_123/BiasAddBiasAddwhile/lstm_cell_123/add:z:02while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_123/splitSplit,while/lstm_cell_123/split/split_dim:output:0$while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_123/SigmoidSigmoid"while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_1Sigmoid"while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_123/mulMul!while/lstm_cell_123/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_123/ReluRelu"while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_123/mul_1Mulwhile/lstm_cell_123/Sigmoid:y:0&while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_123/add_1AddV2while/lstm_cell_123/mul:z:0while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_2Sigmoid"while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_123/Relu_1Reluwhile/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_123/mul_2Mul!while/lstm_cell_123/Sigmoid_2:y:0(while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_123/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_123/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_123/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_123/BiasAdd/ReadVariableOp*^while/lstm_cell_123/MatMul/ReadVariableOp,^while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_123_biasadd_readvariableop_resource5while_lstm_cell_123_biasadd_readvariableop_resource_0"n
4while_lstm_cell_123_matmul_1_readvariableop_resource6while_lstm_cell_123_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_123_matmul_readvariableop_resource4while_lstm_cell_123_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_123/BiasAdd/ReadVariableOp*while/lstm_cell_123/BiasAdd/ReadVariableOp2V
)while/lstm_cell_123/MatMul/ReadVariableOp)while/lstm_cell_123/MatMul/ReadVariableOp2Z
+while/lstm_cell_123/MatMul_1/ReadVariableOp+while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ћ
М
K__inference_sequential_91_layer_call_and_return_conditional_losses_23345996

inputs$
lstm_117_23345663:	–%
lstm_117_23345665:
і– 
lstm_117_23345667:	–%
lstm_118_23345813:
і–%
lstm_118_23345815:
і– 
lstm_118_23345817:	–%
lstm_119_23345965:
і–%
lstm_119_23345967:
і– 
lstm_119_23345969:	–$
dense_89_23345990:	і
dense_89_23345992:
identityИҐ dense_89/StatefulPartitionedCallҐ lstm_117/StatefulPartitionedCallҐ lstm_118/StatefulPartitionedCallҐ lstm_119/StatefulPartitionedCallР
 lstm_117/StatefulPartitionedCallStatefulPartitionedCallinputslstm_117_23345663lstm_117_23345665lstm_117_23345667*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23345662≥
 lstm_118/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0lstm_118_23345813lstm_118_23345815lstm_118_23345817*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23345812ѓ
 lstm_119/StatefulPartitionedCallStatefulPartitionedCall)lstm_118/StatefulPartitionedCall:output:0lstm_119_23345965lstm_119_23345967lstm_119_23345969*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345964в
dropout_72/PartitionedCallPartitionedCall)lstm_119/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_72_layer_call_and_return_conditional_losses_23345977У
 dense_89/StatefulPartitionedCallStatefulPartitionedCall#dropout_72/PartitionedCall:output:0dense_89_23345990dense_89_23345992*
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
F__inference_dense_89_layer_call_and_return_conditional_losses_23345989x
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€“
NoOpNoOp!^dense_89/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall!^lstm_118/StatefulPartitionedCall!^lstm_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall2D
 lstm_118/StatefulPartitionedCall lstm_118/StatefulPartitionedCall2D
 lstm_119/StatefulPartitionedCall lstm_119/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
МL
¶
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345964

inputs@
,lstm_cell_124_matmul_readvariableop_resource:
і–B
.lstm_cell_124_matmul_1_readvariableop_resource:
і–<
-lstm_cell_124_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_124/BiasAdd/ReadVariableOpҐ#lstm_cell_124/MatMul/ReadVariableOpҐ%lstm_cell_124/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_124/MatMul/ReadVariableOpReadVariableOp,lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_124/MatMulMatMulstrided_slice_2:output:0+lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_124/MatMul_1MatMulzeros:output:0-lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_124/addAddV2lstm_cell_124/MatMul:product:0 lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_124/BiasAddBiasAddlstm_cell_124/add:z:0,lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_124/splitSplit&lstm_cell_124/split/split_dim:output:0lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_124/SigmoidSigmoidlstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_1Sigmoidlstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_124/mulMullstm_cell_124/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_124/ReluRelulstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_124/mul_1Mullstm_cell_124/Sigmoid:y:0 lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_124/add_1AddV2lstm_cell_124/mul:z:0lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_2Sigmoidlstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_124/Relu_1Relulstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_124/mul_2Mullstm_cell_124/Sigmoid_2:y:0"lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_124_matmul_readvariableop_resource.lstm_cell_124_matmul_1_readvariableop_resource-lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23345879*
condR
while_cond_23345878*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_124/BiasAdd/ReadVariableOp$^lstm_cell_124/MatMul/ReadVariableOp&^lstm_cell_124/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_124/BiasAdd/ReadVariableOp$lstm_cell_124/BiasAdd/ReadVariableOp2J
#lstm_cell_124/MatMul/ReadVariableOp#lstm_cell_124/MatMul/ReadVariableOp2N
%lstm_cell_124/MatMul_1/ReadVariableOp%lstm_cell_124/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
о
И
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23344526

inputs

states
states_11
matmul_readvariableop_resource:	–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:€€€€€€€€€:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates
√
Ќ
while_cond_23348208
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23348208___redundant_placeholder06
2while_while_cond_23348208___redundant_placeholder16
2while_while_cond_23348208___redundant_placeholder26
2while_while_cond_23348208___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
МL
¶
F__inference_lstm_119_layer_call_and_return_conditional_losses_23346212

inputs@
,lstm_cell_124_matmul_readvariableop_resource:
і–B
.lstm_cell_124_matmul_1_readvariableop_resource:
і–<
-lstm_cell_124_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_124/BiasAdd/ReadVariableOpҐ#lstm_cell_124/MatMul/ReadVariableOpҐ%lstm_cell_124/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_124/MatMul/ReadVariableOpReadVariableOp,lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_124/MatMulMatMulstrided_slice_2:output:0+lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_124/MatMul_1MatMulzeros:output:0-lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_124/addAddV2lstm_cell_124/MatMul:product:0 lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_124/BiasAddBiasAddlstm_cell_124/add:z:0,lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_124/splitSplit&lstm_cell_124/split/split_dim:output:0lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_124/SigmoidSigmoidlstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_1Sigmoidlstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_124/mulMullstm_cell_124/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_124/ReluRelulstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_124/mul_1Mullstm_cell_124/Sigmoid:y:0 lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_124/add_1AddV2lstm_cell_124/mul:z:0lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_2Sigmoidlstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_124/Relu_1Relulstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_124/mul_2Mullstm_cell_124/Sigmoid_2:y:0"lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_124_matmul_readvariableop_resource.lstm_cell_124_matmul_1_readvariableop_resource-lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23346127*
condR
while_cond_23346126*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_124/BiasAdd/ReadVariableOp$^lstm_cell_124/MatMul/ReadVariableOp&^lstm_cell_124/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_124/BiasAdd/ReadVariableOp$lstm_cell_124/BiasAdd/ReadVariableOp2J
#lstm_cell_124/MatMul/ReadVariableOp#lstm_cell_124/MatMul/ReadVariableOp2N
%lstm_cell_124/MatMul_1/ReadVariableOp%lstm_cell_124/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
√
Ќ
while_cond_23348681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23348681___redundant_placeholder06
2while_while_cond_23348681___redundant_placeholder16
2while_while_cond_23348681___redundant_placeholder26
2while_while_cond_23348681___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
£T
и
!__inference__traced_save_23350013
file_prefix.
*savev2_dense_89_kernel_read_readvariableop,
(savev2_dense_89_bias_read_readvariableop<
8savev2_lstm_117_lstm_cell_122_kernel_read_readvariableopF
Bsavev2_lstm_117_lstm_cell_122_recurrent_kernel_read_readvariableop:
6savev2_lstm_117_lstm_cell_122_bias_read_readvariableop<
8savev2_lstm_118_lstm_cell_123_kernel_read_readvariableopF
Bsavev2_lstm_118_lstm_cell_123_recurrent_kernel_read_readvariableop:
6savev2_lstm_118_lstm_cell_123_bias_read_readvariableop<
8savev2_lstm_119_lstm_cell_124_kernel_read_readvariableopF
Bsavev2_lstm_119_lstm_cell_124_recurrent_kernel_read_readvariableop:
6savev2_lstm_119_lstm_cell_124_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopC
?savev2_adam_m_lstm_117_lstm_cell_122_kernel_read_readvariableopC
?savev2_adam_v_lstm_117_lstm_cell_122_kernel_read_readvariableopM
Isavev2_adam_m_lstm_117_lstm_cell_122_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_117_lstm_cell_122_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_117_lstm_cell_122_bias_read_readvariableopA
=savev2_adam_v_lstm_117_lstm_cell_122_bias_read_readvariableopC
?savev2_adam_m_lstm_118_lstm_cell_123_kernel_read_readvariableopC
?savev2_adam_v_lstm_118_lstm_cell_123_kernel_read_readvariableopM
Isavev2_adam_m_lstm_118_lstm_cell_123_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_118_lstm_cell_123_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_118_lstm_cell_123_bias_read_readvariableopA
=savev2_adam_v_lstm_118_lstm_cell_123_bias_read_readvariableopC
?savev2_adam_m_lstm_119_lstm_cell_124_kernel_read_readvariableopC
?savev2_adam_v_lstm_119_lstm_cell_124_kernel_read_readvariableopM
Isavev2_adam_m_lstm_119_lstm_cell_124_recurrent_kernel_read_readvariableopM
Isavev2_adam_v_lstm_119_lstm_cell_124_recurrent_kernel_read_readvariableopA
=savev2_adam_m_lstm_119_lstm_cell_124_bias_read_readvariableopA
=savev2_adam_v_lstm_119_lstm_cell_124_bias_read_readvariableop5
1savev2_adam_m_dense_89_kernel_read_readvariableop5
1savev2_adam_v_dense_89_kernel_read_readvariableop3
/savev2_adam_m_dense_89_bias_read_readvariableop3
/savev2_adam_v_dense_89_bias_read_readvariableop&
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
: О
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*Ј
value≠B™(B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHљ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ё
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_89_kernel_read_readvariableop(savev2_dense_89_bias_read_readvariableop8savev2_lstm_117_lstm_cell_122_kernel_read_readvariableopBsavev2_lstm_117_lstm_cell_122_recurrent_kernel_read_readvariableop6savev2_lstm_117_lstm_cell_122_bias_read_readvariableop8savev2_lstm_118_lstm_cell_123_kernel_read_readvariableopBsavev2_lstm_118_lstm_cell_123_recurrent_kernel_read_readvariableop6savev2_lstm_118_lstm_cell_123_bias_read_readvariableop8savev2_lstm_119_lstm_cell_124_kernel_read_readvariableopBsavev2_lstm_119_lstm_cell_124_recurrent_kernel_read_readvariableop6savev2_lstm_119_lstm_cell_124_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop?savev2_adam_m_lstm_117_lstm_cell_122_kernel_read_readvariableop?savev2_adam_v_lstm_117_lstm_cell_122_kernel_read_readvariableopIsavev2_adam_m_lstm_117_lstm_cell_122_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_117_lstm_cell_122_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_117_lstm_cell_122_bias_read_readvariableop=savev2_adam_v_lstm_117_lstm_cell_122_bias_read_readvariableop?savev2_adam_m_lstm_118_lstm_cell_123_kernel_read_readvariableop?savev2_adam_v_lstm_118_lstm_cell_123_kernel_read_readvariableopIsavev2_adam_m_lstm_118_lstm_cell_123_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_118_lstm_cell_123_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_118_lstm_cell_123_bias_read_readvariableop=savev2_adam_v_lstm_118_lstm_cell_123_bias_read_readvariableop?savev2_adam_m_lstm_119_lstm_cell_124_kernel_read_readvariableop?savev2_adam_v_lstm_119_lstm_cell_124_kernel_read_readvariableopIsavev2_adam_m_lstm_119_lstm_cell_124_recurrent_kernel_read_readvariableopIsavev2_adam_v_lstm_119_lstm_cell_124_recurrent_kernel_read_readvariableop=savev2_adam_m_lstm_119_lstm_cell_124_bias_read_readvariableop=savev2_adam_v_lstm_119_lstm_cell_124_bias_read_readvariableop1savev2_adam_m_dense_89_kernel_read_readvariableop1savev2_adam_v_dense_89_kernel_read_readvariableop/savev2_adam_m_dense_89_bias_read_readvariableop/savev2_adam_v_dense_89_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *6
dtypes,
*2(	Р
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

identity_1Identity_1:output:0*м
_input_shapesЏ
„: :	і::	–:
і–:–:
і–:
і–:–:
і–:
і–:–: : :	–:	–:
і–:
і–:–:–:
і–:
і–:
і–:
і–:–:–:
і–:
і–:
і–:
і–:–:–:	і:	і::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	і: 

_output_shapes
::%!

_output_shapes
:	–:&"
 
_output_shapes
:
і–:!

_output_shapes	
:–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:!

_output_shapes	
:–:&	"
 
_output_shapes
:
і–:&
"
 
_output_shapes
:
і–:!

_output_shapes	
:–:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	–:%!

_output_shapes
:	–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:!

_output_shapes	
:–:!

_output_shapes	
:–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:!

_output_shapes	
:–:!

_output_shapes	
:–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:&"
 
_output_shapes
:
і–:!

_output_shapes	
:–:!

_output_shapes	
:–:% !

_output_shapes
:	і:%!!

_output_shapes
:	і: "
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
з

•
0__inference_sequential_91_layer_call_fn_23346810

inputs
unknown:	–
	unknown_0:
і–
	unknown_1:	–
	unknown_2:
і–
	unknown_3:
і–
	unknown_4:	–
	unknown_5:
і–
	unknown_6:
і–
	unknown_7:	–
	unknown_8:	і
	unknown_9:
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346611o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
√
Ќ
while_cond_23344539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23344539___redundant_placeholder06
2while_while_cond_23344539___redundant_placeholder16
2while_while_cond_23344539___redundant_placeholder26
2while_while_cond_23344539___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ј

Б
lstm_117_while_cond_23347298.
*lstm_117_while_lstm_117_while_loop_counter4
0lstm_117_while_lstm_117_while_maximum_iterations
lstm_117_while_placeholder 
lstm_117_while_placeholder_1 
lstm_117_while_placeholder_2 
lstm_117_while_placeholder_30
,lstm_117_while_less_lstm_117_strided_slice_1H
Dlstm_117_while_lstm_117_while_cond_23347298___redundant_placeholder0H
Dlstm_117_while_lstm_117_while_cond_23347298___redundant_placeholder1H
Dlstm_117_while_lstm_117_while_cond_23347298___redundant_placeholder2H
Dlstm_117_while_lstm_117_while_cond_23347298___redundant_placeholder3
lstm_117_while_identity
Ж
lstm_117/while/LessLesslstm_117_while_placeholder,lstm_117_while_less_lstm_117_strided_slice_1*
T0*
_output_shapes
: ]
lstm_117/while/IdentityIdentitylstm_117/while/Less:z:0*
T0
*
_output_shapes
: ";
lstm_117_while_identity lstm_117/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
МL
¶
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349533

inputs@
,lstm_cell_124_matmul_readvariableop_resource:
і–B
.lstm_cell_124_matmul_1_readvariableop_resource:
і–<
-lstm_cell_124_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_124/BiasAdd/ReadVariableOpҐ#lstm_cell_124/MatMul/ReadVariableOpҐ%lstm_cell_124/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskТ
#lstm_cell_124/MatMul/ReadVariableOpReadVariableOp,lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Ш
lstm_cell_124/MatMulMatMulstrided_slice_2:output:0+lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_124/MatMul_1MatMulzeros:output:0-lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_124/addAddV2lstm_cell_124/MatMul:product:0 lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_124/BiasAddBiasAddlstm_cell_124/add:z:0,lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_124/splitSplit&lstm_cell_124/split/split_dim:output:0lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_124/SigmoidSigmoidlstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_1Sigmoidlstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_124/mulMullstm_cell_124/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_124/ReluRelulstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_124/mul_1Mullstm_cell_124/Sigmoid:y:0 lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_124/add_1AddV2lstm_cell_124/mul:z:0lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_124/Sigmoid_2Sigmoidlstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_124/Relu_1Relulstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_124/mul_2Mullstm_cell_124/Sigmoid_2:y:0"lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ^
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_124_matmul_readvariableop_resource.lstm_cell_124_matmul_1_readvariableop_resource-lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23349448*
condR
while_cond_23349447*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   „
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_124/BiasAdd/ReadVariableOp$^lstm_cell_124/MatMul/ReadVariableOp&^lstm_cell_124/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 2L
$lstm_cell_124/BiasAdd/ReadVariableOp$lstm_cell_124/BiasAdd/ReadVariableOp2J
#lstm_cell_124/MatMul/ReadVariableOp#lstm_cell_124/MatMul/ReadVariableOp2N
%lstm_cell_124/MatMul_1/ReadVariableOp%lstm_cell_124/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
≤U
¬
*sequential_91_lstm_119_while_body_23344367J
Fsequential_91_lstm_119_while_sequential_91_lstm_119_while_loop_counterP
Lsequential_91_lstm_119_while_sequential_91_lstm_119_while_maximum_iterations,
(sequential_91_lstm_119_while_placeholder.
*sequential_91_lstm_119_while_placeholder_1.
*sequential_91_lstm_119_while_placeholder_2.
*sequential_91_lstm_119_while_placeholder_3I
Esequential_91_lstm_119_while_sequential_91_lstm_119_strided_slice_1_0Ж
Бsequential_91_lstm_119_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_119_tensorarrayunstack_tensorlistfromtensor_0_
Ksequential_91_lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0:
і–a
Msequential_91_lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–[
Lsequential_91_lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0:	–)
%sequential_91_lstm_119_while_identity+
'sequential_91_lstm_119_while_identity_1+
'sequential_91_lstm_119_while_identity_2+
'sequential_91_lstm_119_while_identity_3+
'sequential_91_lstm_119_while_identity_4+
'sequential_91_lstm_119_while_identity_5G
Csequential_91_lstm_119_while_sequential_91_lstm_119_strided_slice_1Г
sequential_91_lstm_119_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_119_tensorarrayunstack_tensorlistfromtensor]
Isequential_91_lstm_119_while_lstm_cell_124_matmul_readvariableop_resource:
і–_
Ksequential_91_lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource:
і–Y
Jsequential_91_lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐAsequential_91/lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpҐ@sequential_91/lstm_119/while/lstm_cell_124/MatMul/ReadVariableOpҐBsequential_91/lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOpЯ
Nsequential_91/lstm_119/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Ы
@sequential_91/lstm_119/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemБsequential_91_lstm_119_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_119_tensorarrayunstack_tensorlistfromtensor_0(sequential_91_lstm_119_while_placeholderWsequential_91/lstm_119/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0ќ
@sequential_91/lstm_119/while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOpKsequential_91_lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Б
1sequential_91/lstm_119/while/lstm_cell_124/MatMulMatMulGsequential_91/lstm_119/while/TensorArrayV2Read/TensorListGetItem:item:0Hsequential_91/lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–“
Bsequential_91/lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOpMsequential_91_lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0и
3sequential_91/lstm_119/while/lstm_cell_124/MatMul_1MatMul*sequential_91_lstm_119_while_placeholder_2Jsequential_91/lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ж
.sequential_91/lstm_119/while/lstm_cell_124/addAddV2;sequential_91/lstm_119/while/lstm_cell_124/MatMul:product:0=sequential_91/lstm_119/while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Ћ
Asequential_91/lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOpLsequential_91_lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0п
2sequential_91/lstm_119/while/lstm_cell_124/BiasAddBiasAdd2sequential_91/lstm_119/while/lstm_cell_124/add:z:0Isequential_91/lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–|
:sequential_91/lstm_119/while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
0sequential_91/lstm_119/while/lstm_cell_124/splitSplitCsequential_91/lstm_119/while/lstm_cell_124/split/split_dim:output:0;sequential_91/lstm_119/while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitЂ
2sequential_91/lstm_119/while/lstm_cell_124/SigmoidSigmoid9sequential_91/lstm_119/while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і≠
4sequential_91/lstm_119/while/lstm_cell_124/Sigmoid_1Sigmoid9sequential_91/lstm_119/while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іќ
.sequential_91/lstm_119/while/lstm_cell_124/mulMul8sequential_91/lstm_119/while/lstm_cell_124/Sigmoid_1:y:0*sequential_91_lstm_119_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€і•
/sequential_91/lstm_119/while/lstm_cell_124/ReluRelu9sequential_91/lstm_119/while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іб
0sequential_91/lstm_119/while/lstm_cell_124/mul_1Mul6sequential_91/lstm_119/while/lstm_cell_124/Sigmoid:y:0=sequential_91/lstm_119/while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і÷
0sequential_91/lstm_119/while/lstm_cell_124/add_1AddV22sequential_91/lstm_119/while/lstm_cell_124/mul:z:04sequential_91/lstm_119/while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і≠
4sequential_91/lstm_119/while/lstm_cell_124/Sigmoid_2Sigmoid9sequential_91/lstm_119/while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іҐ
1sequential_91/lstm_119/while/lstm_cell_124/Relu_1Relu4sequential_91/lstm_119/while/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іе
0sequential_91/lstm_119/while/lstm_cell_124/mul_2Mul8sequential_91/lstm_119/while/lstm_cell_124/Sigmoid_2:y:0?sequential_91/lstm_119/while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЙ
Gsequential_91/lstm_119/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B :  
Asequential_91/lstm_119/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem*sequential_91_lstm_119_while_placeholder_1Psequential_91/lstm_119/while/TensorArrayV2Write/TensorListSetItem/index:output:04sequential_91/lstm_119/while/lstm_cell_124/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“d
"sequential_91/lstm_119/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :°
 sequential_91/lstm_119/while/addAddV2(sequential_91_lstm_119_while_placeholder+sequential_91/lstm_119/while/add/y:output:0*
T0*
_output_shapes
: f
$sequential_91/lstm_119/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :√
"sequential_91/lstm_119/while/add_1AddV2Fsequential_91_lstm_119_while_sequential_91_lstm_119_while_loop_counter-sequential_91/lstm_119/while/add_1/y:output:0*
T0*
_output_shapes
: Ю
%sequential_91/lstm_119/while/IdentityIdentity&sequential_91/lstm_119/while/add_1:z:0"^sequential_91/lstm_119/while/NoOp*
T0*
_output_shapes
: ∆
'sequential_91/lstm_119/while/Identity_1IdentityLsequential_91_lstm_119_while_sequential_91_lstm_119_while_maximum_iterations"^sequential_91/lstm_119/while/NoOp*
T0*
_output_shapes
: Ю
'sequential_91/lstm_119/while/Identity_2Identity$sequential_91/lstm_119/while/add:z:0"^sequential_91/lstm_119/while/NoOp*
T0*
_output_shapes
: Ћ
'sequential_91/lstm_119/while/Identity_3IdentityQsequential_91/lstm_119/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^sequential_91/lstm_119/while/NoOp*
T0*
_output_shapes
: ј
'sequential_91/lstm_119/while/Identity_4Identity4sequential_91/lstm_119/while/lstm_cell_124/mul_2:z:0"^sequential_91/lstm_119/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іј
'sequential_91/lstm_119/while/Identity_5Identity4sequential_91/lstm_119/while/lstm_cell_124/add_1:z:0"^sequential_91/lstm_119/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іѓ
!sequential_91/lstm_119/while/NoOpNoOpB^sequential_91/lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpA^sequential_91/lstm_119/while/lstm_cell_124/MatMul/ReadVariableOpC^sequential_91/lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "W
%sequential_91_lstm_119_while_identity.sequential_91/lstm_119/while/Identity:output:0"[
'sequential_91_lstm_119_while_identity_10sequential_91/lstm_119/while/Identity_1:output:0"[
'sequential_91_lstm_119_while_identity_20sequential_91/lstm_119/while/Identity_2:output:0"[
'sequential_91_lstm_119_while_identity_30sequential_91/lstm_119/while/Identity_3:output:0"[
'sequential_91_lstm_119_while_identity_40sequential_91/lstm_119/while/Identity_4:output:0"[
'sequential_91_lstm_119_while_identity_50sequential_91/lstm_119/while/Identity_5:output:0"Ъ
Jsequential_91_lstm_119_while_lstm_cell_124_biasadd_readvariableop_resourceLsequential_91_lstm_119_while_lstm_cell_124_biasadd_readvariableop_resource_0"Ь
Ksequential_91_lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resourceMsequential_91_lstm_119_while_lstm_cell_124_matmul_1_readvariableop_resource_0"Ш
Isequential_91_lstm_119_while_lstm_cell_124_matmul_readvariableop_resourceKsequential_91_lstm_119_while_lstm_cell_124_matmul_readvariableop_resource_0"М
Csequential_91_lstm_119_while_sequential_91_lstm_119_strided_slice_1Esequential_91_lstm_119_while_sequential_91_lstm_119_strided_slice_1_0"Е
sequential_91_lstm_119_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_119_tensorarrayunstack_tensorlistfromtensorБsequential_91_lstm_119_while_tensorarrayv2read_tensorlistgetitem_sequential_91_lstm_119_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Ж
Asequential_91/lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOpAsequential_91/lstm_119/while/lstm_cell_124/BiasAdd/ReadVariableOp2Д
@sequential_91/lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp@sequential_91/lstm_119/while/lstm_cell_124/MatMul/ReadVariableOp2И
Bsequential_91/lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOpBsequential_91/lstm_119/while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
Х

g
H__inference_dropout_72_layer_call_and_return_conditional_losses_23346051

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€і*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€іT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€і"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€і:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
√
Ќ
while_cond_23345878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23345878___redundant_placeholder06
2while_while_cond_23345878___redundant_placeholder16
2while_while_cond_23345878___redundant_placeholder26
2while_while_cond_23345878___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
√
Ќ
while_cond_23345577
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23345577___redundant_placeholder06
2while_while_cond_23345577___redundant_placeholder16
2while_while_cond_23345577___redundant_placeholder26
2while_while_cond_23345577___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
≤
ї
+__inference_lstm_117_layer_call_fn_23347699
inputs_0
unknown:	–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23344800}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і`
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
и8
Ё
while_body_23346458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_122_matmul_readvariableop_resource_0:	–J
6while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_122_matmul_readvariableop_resource:	–H
4while_lstm_cell_122_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_122/BiasAdd/ReadVariableOpҐ)while/lstm_cell_122/MatMul/ReadVariableOpҐ+while/lstm_cell_122/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Я
)while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0Љ
while/lstm_cell_122/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_122/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_122/addAddV2$while/lstm_cell_122/MatMul:product:0&while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_122/BiasAddBiasAddwhile/lstm_cell_122/add:z:02while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_122/splitSplit,while/lstm_cell_122/split/split_dim:output:0$while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_122/SigmoidSigmoid"while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_1Sigmoid"while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_122/mulMul!while/lstm_cell_122/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_122/ReluRelu"while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_122/mul_1Mulwhile/lstm_cell_122/Sigmoid:y:0&while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_122/add_1AddV2while/lstm_cell_122/mul:z:0while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_2Sigmoid"while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_122/Relu_1Reluwhile/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_122/mul_2Mul!while/lstm_cell_122/Sigmoid_2:y:0(while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_122/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_122/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_122/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_122/BiasAdd/ReadVariableOp*^while/lstm_cell_122/MatMul/ReadVariableOp,^while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_122_biasadd_readvariableop_resource5while_lstm_cell_122_biasadd_readvariableop_resource_0"n
4while_lstm_cell_122_matmul_1_readvariableop_resource6while_lstm_cell_122_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_122_matmul_readvariableop_resource4while_lstm_cell_122_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_122/BiasAdd/ReadVariableOp*while/lstm_cell_122/BiasAdd/ReadVariableOp2V
)while/lstm_cell_122/MatMul/ReadVariableOp)while/lstm_cell_122/MatMul/ReadVariableOp2Z
+while/lstm_cell_122/MatMul_1/ReadVariableOp+while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
»
Щ
*sequential_91_lstm_118_while_cond_23344226J
Fsequential_91_lstm_118_while_sequential_91_lstm_118_while_loop_counterP
Lsequential_91_lstm_118_while_sequential_91_lstm_118_while_maximum_iterations,
(sequential_91_lstm_118_while_placeholder.
*sequential_91_lstm_118_while_placeholder_1.
*sequential_91_lstm_118_while_placeholder_2.
*sequential_91_lstm_118_while_placeholder_3L
Hsequential_91_lstm_118_while_less_sequential_91_lstm_118_strided_slice_1d
`sequential_91_lstm_118_while_sequential_91_lstm_118_while_cond_23344226___redundant_placeholder0d
`sequential_91_lstm_118_while_sequential_91_lstm_118_while_cond_23344226___redundant_placeholder1d
`sequential_91_lstm_118_while_sequential_91_lstm_118_while_cond_23344226___redundant_placeholder2d
`sequential_91_lstm_118_while_sequential_91_lstm_118_while_cond_23344226___redundant_placeholder3)
%sequential_91_lstm_118_while_identity
Њ
!sequential_91/lstm_118/while/LessLess(sequential_91_lstm_118_while_placeholderHsequential_91_lstm_118_while_less_sequential_91_lstm_118_strided_slice_1*
T0*
_output_shapes
: y
%sequential_91/lstm_118/while/IdentityIdentity%sequential_91/lstm_118/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_91_lstm_118_while_identity.sequential_91/lstm_118/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
 $
ь
while_body_23345241
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_lstm_cell_124_23345265_0:
і–2
while_lstm_cell_124_23345267_0:
і–-
while_lstm_cell_124_23345269_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_lstm_cell_124_23345265:
і–0
while_lstm_cell_124_23345267:
і–+
while_lstm_cell_124_23345269:	–ИҐ+while/lstm_cell_124/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0√
+while/lstm_cell_124/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_124_23345265_0while_lstm_cell_124_23345267_0while_lstm_cell_124_23345269_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23345226r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Е
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:04while/lstm_cell_124/StatefulPartitionedCall:output:0*
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
: Т
while/Identity_4Identity4while/lstm_cell_124/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іТ
while/Identity_5Identity4while/lstm_cell_124/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іz

while/NoOpNoOp,^while/lstm_cell_124/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0">
while_lstm_cell_124_23345265while_lstm_cell_124_23345265_0">
while_lstm_cell_124_23345267while_lstm_cell_124_23345267_0">
while_lstm_cell_124_23345269while_lstm_cell_124_23345269_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2Z
+while/lstm_cell_124/StatefulPartitionedCall+while/lstm_cell_124/StatefulPartitionedCall: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
И
є
+__inference_lstm_117_layer_call_fn_23347710

inputs
unknown:	–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23345662t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і`
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
√
Ќ
while_cond_23349302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23349302___redundant_placeholder06
2while_while_cond_23349302___redundant_placeholder16
2while_while_cond_23349302___redundant_placeholder26
2while_while_cond_23349302___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
И:
я
while_body_23346127
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_124_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_124_matmul_readvariableop_resource:
і–H
4while_lstm_cell_124_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_124/BiasAdd/ReadVariableOpҐ)while/lstm_cell_124/MatMul/ReadVariableOpҐ+while/lstm_cell_124/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_124/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_124/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_124/addAddV2$while/lstm_cell_124/MatMul:product:0&while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_124/BiasAddBiasAddwhile/lstm_cell_124/add:z:02while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_124/splitSplit,while/lstm_cell_124/split/split_dim:output:0$while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_124/SigmoidSigmoid"while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_1Sigmoid"while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_124/mulMul!while/lstm_cell_124/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_124/ReluRelu"while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_124/mul_1Mulwhile/lstm_cell_124/Sigmoid:y:0&while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_124/add_1AddV2while/lstm_cell_124/mul:z:0while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_2Sigmoid"while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_124/Relu_1Reluwhile/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_124/mul_2Mul!while/lstm_cell_124/Sigmoid_2:y:0(while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_124/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_124/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_124/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_124/BiasAdd/ReadVariableOp*^while/lstm_cell_124/MatMul/ReadVariableOp,^while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_124_biasadd_readvariableop_resource5while_lstm_cell_124_biasadd_readvariableop_resource_0"n
4while_lstm_cell_124_matmul_1_readvariableop_resource6while_lstm_cell_124_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_124_matmul_readvariableop_resource4while_lstm_cell_124_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_124/BiasAdd/ReadVariableOp*while/lstm_cell_124/BiasAdd/ReadVariableOp2V
)while/lstm_cell_124/MatMul/ReadVariableOp)while/lstm_cell_124/MatMul/ReadVariableOp2Z
+while/lstm_cell_124/MatMul_1/ReadVariableOp+while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
√
Ќ
while_cond_23345080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23345080___redundant_placeholder06
2while_while_cond_23345080___redundant_placeholder16
2while_while_cond_23345080___redundant_placeholder26
2while_while_cond_23345080___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
√
Ќ
while_cond_23345240
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23345240___redundant_placeholder06
2while_while_cond_23345240___redundant_placeholder16
2while_while_cond_23345240___redundant_placeholder26
2while_while_cond_23345240___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
√
Ќ
while_cond_23348538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23348538___redundant_placeholder06
2while_while_cond_23348538___redundant_placeholder16
2while_while_cond_23348538___redundant_placeholder26
2while_while_cond_23348538___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
Л
Ї
+__inference_lstm_118_layer_call_fn_23348337

inputs
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23346377t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
АK
•
F__inference_lstm_117_layer_call_and_return_conditional_losses_23345662

inputs?
,lstm_cell_122_matmul_readvariableop_resource:	–B
.lstm_cell_122_matmul_1_readvariableop_resource:
і–<
-lstm_cell_122_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_122/BiasAdd/ReadVariableOpҐ#lstm_cell_122/MatMul/ReadVariableOpҐ%lstm_cell_122/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskС
#lstm_cell_122/MatMul/ReadVariableOpReadVariableOp,lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0Ш
lstm_cell_122/MatMulMatMulstrided_slice_2:output:0+lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_122/MatMul_1MatMulzeros:output:0-lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_122/addAddV2lstm_cell_122/MatMul:product:0 lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_122/BiasAddBiasAddlstm_cell_122/add:z:0,lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_122/splitSplit&lstm_cell_122/split/split_dim:output:0lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_122/SigmoidSigmoidlstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_1Sigmoidlstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_122/mulMullstm_cell_122/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_122/ReluRelulstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_122/mul_1Mullstm_cell_122/Sigmoid:y:0 lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_122/add_1AddV2lstm_cell_122/mul:z:0lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_2Sigmoidlstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_122/Relu_1Relulstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_122/mul_2Mullstm_cell_122/Sigmoid_2:y:0"lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_122_matmul_readvariableop_resource.lstm_cell_122_matmul_1_readvariableop_resource-lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23345578*
condR
while_cond_23345577*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_122/BiasAdd/ReadVariableOp$^lstm_cell_122/MatMul/ReadVariableOp&^lstm_cell_122/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2L
$lstm_cell_122/BiasAdd/ReadVariableOp$lstm_cell_122/BiasAdd/ReadVariableOp2J
#lstm_cell_122/MatMul/ReadVariableOp#lstm_cell_122/MatMul/ReadVariableOp2N
%lstm_cell_122/MatMul_1/ReadVariableOp%lstm_cell_122/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
√
Ќ
while_cond_23345433
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23345433___redundant_placeholder06
2while_while_cond_23345433___redundant_placeholder16
2while_while_cond_23345433___redundant_placeholder26
2while_while_cond_23345433___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
√
Ќ
while_cond_23344889
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23344889___redundant_placeholder06
2while_while_cond_23344889___redundant_placeholder16
2while_while_cond_23344889___redundant_placeholder26
2while_while_cond_23344889___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
т
Й
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23344876

inputs

states
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates
Г
Ї
+__inference_lstm_119_layer_call_fn_23348953

inputs
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23346212p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
И:
я
while_body_23345879
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_124_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_124_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_124_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_124_matmul_readvariableop_resource:
і–H
4while_lstm_cell_124_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_124_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_124/BiasAdd/ReadVariableOpҐ)while/lstm_cell_124/MatMul/ReadVariableOpҐ+while/lstm_cell_124/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_124_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_124/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_124_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_124/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_124/addAddV2$while/lstm_cell_124/MatMul:product:0&while/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_124_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_124/BiasAddBiasAddwhile/lstm_cell_124/add:z:02while/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_124/splitSplit,while/lstm_cell_124/split/split_dim:output:0$while/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_124/SigmoidSigmoid"while/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_1Sigmoid"while/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_124/mulMul!while/lstm_cell_124/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_124/ReluRelu"while/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_124/mul_1Mulwhile/lstm_cell_124/Sigmoid:y:0&while/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_124/add_1AddV2while/lstm_cell_124/mul:z:0while/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_124/Sigmoid_2Sigmoid"while/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_124/Relu_1Reluwhile/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_124/mul_2Mul!while/lstm_cell_124/Sigmoid_2:y:0(while/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : о
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_124/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_124/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_124/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_124/BiasAdd/ReadVariableOp*^while/lstm_cell_124/MatMul/ReadVariableOp,^while/lstm_cell_124/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_124_biasadd_readvariableop_resource5while_lstm_cell_124_biasadd_readvariableop_resource_0"n
4while_lstm_cell_124_matmul_1_readvariableop_resource6while_lstm_cell_124_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_124_matmul_readvariableop_resource4while_lstm_cell_124_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_124/BiasAdd/ReadVariableOp*while/lstm_cell_124/BiasAdd/ReadVariableOp2V
)while/lstm_cell_124/MatMul/ReadVariableOp)while/lstm_cell_124/MatMul/ReadVariableOp2Z
+while/lstm_cell_124/MatMul_1/ReadVariableOp+while/lstm_cell_124/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
д
Ф
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346694
lstm_117_input$
lstm_117_23346666:	–%
lstm_117_23346668:
і– 
lstm_117_23346670:	–%
lstm_118_23346673:
і–%
lstm_118_23346675:
і– 
lstm_118_23346677:	–%
lstm_119_23346680:
і–%
lstm_119_23346682:
і– 
lstm_119_23346684:	–$
dense_89_23346688:	і
dense_89_23346690:
identityИҐ dense_89/StatefulPartitionedCallҐ lstm_117/StatefulPartitionedCallҐ lstm_118/StatefulPartitionedCallҐ lstm_119/StatefulPartitionedCallШ
 lstm_117/StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputlstm_117_23346666lstm_117_23346668lstm_117_23346670*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_117_layer_call_and_return_conditional_losses_23345662≥
 lstm_118/StatefulPartitionedCallStatefulPartitionedCall)lstm_117/StatefulPartitionedCall:output:0lstm_118_23346673lstm_118_23346675lstm_118_23346677*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_118_layer_call_and_return_conditional_losses_23345812ѓ
 lstm_119/StatefulPartitionedCallStatefulPartitionedCall)lstm_118/StatefulPartitionedCall:output:0lstm_119_23346680lstm_119_23346682lstm_119_23346684*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345964в
dropout_72/PartitionedCallPartitionedCall)lstm_119/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_72_layer_call_and_return_conditional_losses_23345977У
 dense_89/StatefulPartitionedCallStatefulPartitionedCall#dropout_72/PartitionedCall:output:0dense_89_23346688dense_89_23346690*
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
F__inference_dense_89_layer_call_and_return_conditional_losses_23345989x
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€“
NoOpNoOp!^dense_89/StatefulPartitionedCall!^lstm_117/StatefulPartitionedCall!^lstm_118/StatefulPartitionedCall!^lstm_119/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall2D
 lstm_117/StatefulPartitionedCall lstm_117/StatefulPartitionedCall2D
 lstm_118/StatefulPartitionedCall lstm_118/StatefulPartitionedCall2D
 lstm_119/StatefulPartitionedCall lstm_119/StatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€
(
_user_specified_namelstm_117_input
ЊK
І
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348007
inputs_0?
,lstm_cell_122_matmul_readvariableop_resource:	–B
.lstm_cell_122_matmul_1_readvariableop_resource:
і–<
-lstm_cell_122_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_122/BiasAdd/ReadVariableOpҐ#lstm_cell_122/MatMul/ReadVariableOpҐ%lstm_cell_122/MatMul_1/ReadVariableOpҐwhile=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskС
#lstm_cell_122/MatMul/ReadVariableOpReadVariableOp,lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0Ш
lstm_cell_122/MatMulMatMulstrided_slice_2:output:0+lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_122/MatMul_1MatMulzeros:output:0-lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_122/addAddV2lstm_cell_122/MatMul:product:0 lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_122/BiasAddBiasAddlstm_cell_122/add:z:0,lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_122/splitSplit&lstm_cell_122/split/split_dim:output:0lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_122/SigmoidSigmoidlstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_1Sigmoidlstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_122/mulMullstm_cell_122/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_122/ReluRelulstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_122/mul_1Mullstm_cell_122/Sigmoid:y:0 lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_122/add_1AddV2lstm_cell_122/mul:z:0lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_2Sigmoidlstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_122/Relu_1Relulstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_122/mul_2Mullstm_cell_122/Sigmoid_2:y:0"lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_122_matmul_readvariableop_resource.lstm_cell_122_matmul_1_readvariableop_resource-lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23347923*
condR
while_cond_23347922*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і√
NoOpNoOp%^lstm_cell_122/BiasAdd/ReadVariableOp$^lstm_cell_122/MatMul/ReadVariableOp&^lstm_cell_122/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2L
$lstm_cell_122/BiasAdd/ReadVariableOp$lstm_cell_122/BiasAdd/ReadVariableOp2J
#lstm_cell_122/MatMul/ReadVariableOp#lstm_cell_122/MatMul/ReadVariableOp2N
%lstm_cell_122/MatMul_1/ReadVariableOp%lstm_cell_122/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
м8
я
while_body_23346293
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_123_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_123_matmul_readvariableop_resource:
і–H
4while_lstm_cell_123_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_123/BiasAdd/ReadVariableOpҐ)while/lstm_cell_123/MatMul/ReadVariableOpҐ+while/lstm_cell_123/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_123/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_123/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_123/addAddV2$while/lstm_cell_123/MatMul:product:0&while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_123/BiasAddBiasAddwhile/lstm_cell_123/add:z:02while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_123/splitSplit,while/lstm_cell_123/split/split_dim:output:0$while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_123/SigmoidSigmoid"while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_1Sigmoid"while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_123/mulMul!while/lstm_cell_123/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_123/ReluRelu"while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_123/mul_1Mulwhile/lstm_cell_123/Sigmoid:y:0&while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_123/add_1AddV2while/lstm_cell_123/mul:z:0while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_2Sigmoid"while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_123/Relu_1Reluwhile/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_123/mul_2Mul!while/lstm_cell_123/Sigmoid_2:y:0(while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_123/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_123/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_123/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_123/BiasAdd/ReadVariableOp*^while/lstm_cell_123/MatMul/ReadVariableOp,^while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_123_biasadd_readvariableop_resource5while_lstm_cell_123_biasadd_readvariableop_resource_0"n
4while_lstm_cell_123_matmul_1_readvariableop_resource6while_lstm_cell_123_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_123_matmul_readvariableop_resource4while_lstm_cell_123_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_123/BiasAdd/ReadVariableOp*while/lstm_cell_123/BiasAdd/ReadVariableOp2V
)while/lstm_cell_123/MatMul/ReadVariableOp)while/lstm_cell_123/MatMul/ReadVariableOp2Z
+while/lstm_cell_123/MatMul_1/ReadVariableOp+while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
Г
Ї
+__inference_lstm_119_layer_call_fn_23348942

inputs
unknown:
і–
	unknown_0:
і–
	unknown_1:	–
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_119_layer_call_and_return_conditional_losses_23345964p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:€€€€€€€€€і: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs
—8
Ф
F__inference_lstm_118_layer_call_and_return_conditional_losses_23345150

inputs*
lstm_cell_123_23345068:
і–*
lstm_cell_123_23345070:
і–%
lstm_cell_123_23345072:	–
identityИҐ%lstm_cell_123/StatefulPartitionedCallҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іD
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
valueB"€€€€і   а
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
valueB:к
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskЕ
%lstm_cell_123/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_123_23345068lstm_cell_123_23345070lstm_cell_123_23345072*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *T
fORM
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23345022n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : »
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_123_23345068lstm_cell_123_23345070lstm_cell_123_23345072*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23345081*
condR
while_cond_23345080*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€іv
NoOpNoOp&^lstm_cell_123/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€і: : : 2N
%lstm_cell_123/StatefulPartitionedCall%lstm_cell_123/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і
 
_user_specified_nameinputs
√
Ќ
while_cond_23349012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_23349012___redundant_placeholder06
2while_while_cond_23349012___redundant_placeholder16
2while_while_cond_23349012___redundant_placeholder26
2while_while_cond_23349012___redundant_placeholder3
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
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
ЊK
І
F__inference_lstm_117_layer_call_and_return_conditional_losses_23347864
inputs_0?
,lstm_cell_122_matmul_readvariableop_resource:	–B
.lstm_cell_122_matmul_1_readvariableop_resource:
і–<
-lstm_cell_122_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_122/BiasAdd/ReadVariableOpҐ#lstm_cell_122/MatMul/ReadVariableOpҐ%lstm_cell_122/MatMul_1/ReadVariableOpҐwhile=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskС
#lstm_cell_122/MatMul/ReadVariableOpReadVariableOp,lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0Ш
lstm_cell_122/MatMulMatMulstrided_slice_2:output:0+lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_122/MatMul_1MatMulzeros:output:0-lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_122/addAddV2lstm_cell_122/MatMul:product:0 lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_122/BiasAddBiasAddlstm_cell_122/add:z:0,lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_122/splitSplit&lstm_cell_122/split/split_dim:output:0lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_122/SigmoidSigmoidlstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_1Sigmoidlstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_122/mulMullstm_cell_122/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_122/ReluRelulstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_122/mul_1Mullstm_cell_122/Sigmoid:y:0 lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_122/add_1AddV2lstm_cell_122/mul:z:0lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_2Sigmoidlstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_122/Relu_1Relulstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_122/mul_2Mullstm_cell_122/Sigmoid_2:y:0"lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_122_matmul_readvariableop_resource.lstm_cell_122_matmul_1_readvariableop_resource-lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23347780*
condR
while_cond_23347779*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          †
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€і√
NoOpNoOp%^lstm_cell_122/BiasAdd/ReadVariableOp$^lstm_cell_122/MatMul/ReadVariableOp&^lstm_cell_122/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€: : : 2L
$lstm_cell_122/BiasAdd/ReadVariableOp$lstm_cell_122/BiasAdd/ReadVariableOp2J
#lstm_cell_122/MatMul/ReadVariableOp#lstm_cell_122/MatMul/ReadVariableOp2N
%lstm_cell_122/MatMul_1/ReadVariableOp%lstm_cell_122/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
"
_user_specified_name
inputs_0
и8
Ё
while_body_23347780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
4while_lstm_cell_122_matmul_readvariableop_resource_0:	–J
6while_lstm_cell_122_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_122_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
2while_lstm_cell_122_matmul_readvariableop_resource:	–H
4while_lstm_cell_122_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_122_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_122/BiasAdd/ReadVariableOpҐ)while/lstm_cell_122/MatMul/ReadVariableOpҐ+while/lstm_cell_122/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€*
element_dtype0Я
)while/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_122_matmul_readvariableop_resource_0*
_output_shapes
:	–*
dtype0Љ
while/lstm_cell_122/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_122_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_122/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_122/addAddV2$while/lstm_cell_122/MatMul:product:0&while/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_122_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_122/BiasAddBiasAddwhile/lstm_cell_122/add:z:02while/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_122/splitSplit,while/lstm_cell_122/split/split_dim:output:0$while/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_122/SigmoidSigmoid"while/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_1Sigmoid"while/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_122/mulMul!while/lstm_cell_122/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_122/ReluRelu"while/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_122/mul_1Mulwhile/lstm_cell_122/Sigmoid:y:0&while/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_122/add_1AddV2while/lstm_cell_122/mul:z:0while/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_122/Sigmoid_2Sigmoid"while/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_122/Relu_1Reluwhile/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_122/mul_2Mul!while/lstm_cell_122/Sigmoid_2:y:0(while/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_122/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_122/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_122/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_122/BiasAdd/ReadVariableOp*^while/lstm_cell_122/MatMul/ReadVariableOp,^while/lstm_cell_122/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_122_biasadd_readvariableop_resource5while_lstm_cell_122_biasadd_readvariableop_resource_0"n
4while_lstm_cell_122_matmul_1_readvariableop_resource6while_lstm_cell_122_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_122_matmul_readvariableop_resource4while_lstm_cell_122_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_122/BiasAdd/ReadVariableOp*while/lstm_cell_122/BiasAdd/ReadVariableOp2V
)while/lstm_cell_122/MatMul/ReadVariableOp)while/lstm_cell_122/MatMul/ReadVariableOp2Z
+while/lstm_cell_122/MatMul_1/ReadVariableOp+while/lstm_cell_122/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
ЉC
€

lstm_118_while_body_23347008.
*lstm_118_while_lstm_118_while_loop_counter4
0lstm_118_while_lstm_118_while_maximum_iterations
lstm_118_while_placeholder 
lstm_118_while_placeholder_1 
lstm_118_while_placeholder_2 
lstm_118_while_placeholder_3-
)lstm_118_while_lstm_118_strided_slice_1_0i
elstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensor_0Q
=lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0:
і–S
?lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–M
>lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
lstm_118_while_identity
lstm_118_while_identity_1
lstm_118_while_identity_2
lstm_118_while_identity_3
lstm_118_while_identity_4
lstm_118_while_identity_5+
'lstm_118_while_lstm_118_strided_slice_1g
clstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensorO
;lstm_118_while_lstm_cell_123_matmul_readvariableop_resource:
і–Q
=lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource:
і–K
<lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpҐ2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOpҐ4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOpС
@lstm_118/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ‘
2lstm_118/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemelstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensor_0lstm_118_while_placeholderIlstm_118/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0≤
2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp=lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0„
#lstm_118/while/lstm_cell_123/MatMulMatMul9lstm_118/while/TensorArrayV2Read/TensorListGetItem:item:0:lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–ґ
4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp?lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Њ
%lstm_118/while/lstm_cell_123/MatMul_1MatMullstm_118_while_placeholder_2<lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Љ
 lstm_118/while/lstm_cell_123/addAddV2-lstm_118/while/lstm_cell_123/MatMul:product:0/lstm_118/while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–ѓ
3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp>lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0≈
$lstm_118/while/lstm_cell_123/BiasAddBiasAdd$lstm_118/while/lstm_cell_123/add:z:0;lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–n
,lstm_118/while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :С
"lstm_118/while/lstm_cell_123/splitSplit5lstm_118/while/lstm_cell_123/split/split_dim:output:0-lstm_118/while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitП
$lstm_118/while/lstm_cell_123/SigmoidSigmoid+lstm_118/while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_118/while/lstm_cell_123/Sigmoid_1Sigmoid+lstm_118/while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€і§
 lstm_118/while/lstm_cell_123/mulMul*lstm_118/while/lstm_cell_123/Sigmoid_1:y:0lstm_118_while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іЙ
!lstm_118/while/lstm_cell_123/ReluRelu+lstm_118/while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЈ
"lstm_118/while/lstm_cell_123/mul_1Mul(lstm_118/while/lstm_cell_123/Sigmoid:y:0/lstm_118/while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іђ
"lstm_118/while/lstm_cell_123/add_1AddV2$lstm_118/while/lstm_cell_123/mul:z:0&lstm_118/while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іС
&lstm_118/while/lstm_cell_123/Sigmoid_2Sigmoid+lstm_118/while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іЖ
#lstm_118/while/lstm_cell_123/Relu_1Relu&lstm_118/while/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€ії
"lstm_118/while/lstm_cell_123/mul_2Mul*lstm_118/while/lstm_cell_123/Sigmoid_2:y:01lstm_118/while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€ік
3lstm_118/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_118_while_placeholder_1lstm_118_while_placeholder&lstm_118/while/lstm_cell_123/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“V
lstm_118/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :w
lstm_118/while/addAddV2lstm_118_while_placeholderlstm_118/while/add/y:output:0*
T0*
_output_shapes
: X
lstm_118/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
lstm_118/while/add_1AddV2*lstm_118_while_lstm_118_while_loop_counterlstm_118/while/add_1/y:output:0*
T0*
_output_shapes
: t
lstm_118/while/IdentityIdentitylstm_118/while/add_1:z:0^lstm_118/while/NoOp*
T0*
_output_shapes
: О
lstm_118/while/Identity_1Identity0lstm_118_while_lstm_118_while_maximum_iterations^lstm_118/while/NoOp*
T0*
_output_shapes
: t
lstm_118/while/Identity_2Identitylstm_118/while/add:z:0^lstm_118/while/NoOp*
T0*
_output_shapes
: °
lstm_118/while/Identity_3IdentityClstm_118/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_118/while/NoOp*
T0*
_output_shapes
: Ц
lstm_118/while/Identity_4Identity&lstm_118/while/lstm_cell_123/mul_2:z:0^lstm_118/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іЦ
lstm_118/while/Identity_5Identity&lstm_118/while/lstm_cell_123/add_1:z:0^lstm_118/while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€іч
lstm_118/while/NoOpNoOp4^lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp3^lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp5^lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ";
lstm_118_while_identity lstm_118/while/Identity:output:0"?
lstm_118_while_identity_1"lstm_118/while/Identity_1:output:0"?
lstm_118_while_identity_2"lstm_118/while/Identity_2:output:0"?
lstm_118_while_identity_3"lstm_118/while/Identity_3:output:0"?
lstm_118_while_identity_4"lstm_118/while/Identity_4:output:0"?
lstm_118_while_identity_5"lstm_118/while/Identity_5:output:0"T
'lstm_118_while_lstm_118_strided_slice_1)lstm_118_while_lstm_118_strided_slice_1_0"~
<lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource>lstm_118_while_lstm_cell_123_biasadd_readvariableop_resource_0"А
=lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource?lstm_118_while_lstm_cell_123_matmul_1_readvariableop_resource_0"|
;lstm_118_while_lstm_cell_123_matmul_readvariableop_resource=lstm_118_while_lstm_cell_123_matmul_readvariableop_resource_0"ћ
clstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensorelstm_118_while_tensorarrayv2read_tensorlistgetitem_lstm_118_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2j
3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp3lstm_118/while/lstm_cell_123/BiasAdd/ReadVariableOp2h
2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp2lstm_118/while/lstm_cell_123/MatMul/ReadVariableOp2l
4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp4lstm_118/while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
»
Щ
*sequential_91_lstm_119_while_cond_23344366J
Fsequential_91_lstm_119_while_sequential_91_lstm_119_while_loop_counterP
Lsequential_91_lstm_119_while_sequential_91_lstm_119_while_maximum_iterations,
(sequential_91_lstm_119_while_placeholder.
*sequential_91_lstm_119_while_placeholder_1.
*sequential_91_lstm_119_while_placeholder_2.
*sequential_91_lstm_119_while_placeholder_3L
Hsequential_91_lstm_119_while_less_sequential_91_lstm_119_strided_slice_1d
`sequential_91_lstm_119_while_sequential_91_lstm_119_while_cond_23344366___redundant_placeholder0d
`sequential_91_lstm_119_while_sequential_91_lstm_119_while_cond_23344366___redundant_placeholder1d
`sequential_91_lstm_119_while_sequential_91_lstm_119_while_cond_23344366___redundant_placeholder2d
`sequential_91_lstm_119_while_sequential_91_lstm_119_while_cond_23344366___redundant_placeholder3)
%sequential_91_lstm_119_while_identity
Њ
!sequential_91/lstm_119/while/LessLess(sequential_91_lstm_119_while_placeholderHsequential_91_lstm_119_while_less_sequential_91_lstm_119_strided_slice_1*
T0*
_output_shapes
: y
%sequential_91/lstm_119/while/IdentityIdentity%sequential_91/lstm_119/while/Less:z:0*
T0
*
_output_shapes
: "W
%sequential_91_lstm_119_while_identity.sequential_91/lstm_119/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :€€€€€€€€€і:€€€€€€€€€і: ::::: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
:
м8
я
while_body_23348682
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0H
4while_lstm_cell_123_matmul_readvariableop_resource_0:
і–J
6while_lstm_cell_123_matmul_1_readvariableop_resource_0:
і–D
5while_lstm_cell_123_biasadd_readvariableop_resource_0:	–
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorF
2while_lstm_cell_123_matmul_readvariableop_resource:
і–H
4while_lstm_cell_123_matmul_1_readvariableop_resource:
і–B
3while_lstm_cell_123_biasadd_readvariableop_resource:	–ИҐ*while/lstm_cell_123/BiasAdd/ReadVariableOpҐ)while/lstm_cell_123/MatMul/ReadVariableOpҐ+while/lstm_cell_123/MatMul_1/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:€€€€€€€€€і*
element_dtype0†
)while/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp4while_lstm_cell_123_matmul_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0Љ
while/lstm_cell_123/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:01while/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–§
+while/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp6while_lstm_cell_123_matmul_1_readvariableop_resource_0* 
_output_shapes
:
і–*
dtype0£
while/lstm_cell_123/MatMul_1MatMulwhile_placeholder_23while/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–°
while/lstm_cell_123/addAddV2$while/lstm_cell_123/MatMul:product:0&while/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–Э
*while/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp5while_lstm_cell_123_biasadd_readvariableop_resource_0*
_output_shapes	
:–*
dtype0™
while/lstm_cell_123/BiasAddBiasAddwhile/lstm_cell_123/add:z:02while/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
#while/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ц
while/lstm_cell_123/splitSplit,while/lstm_cell_123/split/split_dim:output:0$while/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_split}
while/lstm_cell_123/SigmoidSigmoid"while/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_1Sigmoid"while/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іЙ
while/lstm_cell_123/mulMul!while/lstm_cell_123/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:€€€€€€€€€іw
while/lstm_cell_123/ReluRelu"while/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іЬ
while/lstm_cell_123/mul_1Mulwhile/lstm_cell_123/Sigmoid:y:0&while/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іС
while/lstm_cell_123/add_1AddV2while/lstm_cell_123/mul:z:0while/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і
while/lstm_cell_123/Sigmoid_2Sigmoid"while/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іt
while/lstm_cell_123/Relu_1Reluwhile/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і†
while/lstm_cell_123/mul_2Mul!while/lstm_cell_123/Sigmoid_2:y:0(while/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і∆
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_123/mul_2:z:0*
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
: {
while/Identity_4Identitywhile/lstm_cell_123/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і{
while/Identity_5Identitywhile/lstm_cell_123/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:€€€€€€€€€і”

while/NoOpNoOp+^while/lstm_cell_123/BiasAdd/ReadVariableOp*^while/lstm_cell_123/MatMul/ReadVariableOp,^while/lstm_cell_123/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"l
3while_lstm_cell_123_biasadd_readvariableop_resource5while_lstm_cell_123_biasadd_readvariableop_resource_0"n
4while_lstm_cell_123_matmul_1_readvariableop_resource6while_lstm_cell_123_matmul_1_readvariableop_resource_0"j
2while_lstm_cell_123_matmul_readvariableop_resource4while_lstm_cell_123_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : 2X
*while/lstm_cell_123/BiasAdd/ReadVariableOp*while/lstm_cell_123/BiasAdd/ReadVariableOp2V
)while/lstm_cell_123/MatMul/ReadVariableOp)while/lstm_cell_123/MatMul/ReadVariableOp2Z
+while/lstm_cell_123/MatMul_1/ReadVariableOp+while/lstm_cell_123/MatMul_1/ReadVariableOp: 
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
:€€€€€€€€€і:.*
(
_output_shapes
:€€€€€€€€€і:

_output_shapes
: :

_output_shapes
: 
т
Й
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23345022

inputs

states
states_12
matmul_readvariableop_resource:
і–4
 matmul_1_readvariableop_resource:
і–.
biasadd_readvariableop_resource:	–
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ї
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:€€€€€€€€€іV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:€€€€€€€€€іO
ReluRelusplit:output:2*
T0*(
_output_shapes
:€€€€€€€€€і`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:€€€€€€€€€іL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€і[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€іС
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates:PL
(
_output_shapes
:€€€€€€€€€і
 
_user_specified_namestates
њМ
†
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347677

inputsH
5lstm_117_lstm_cell_122_matmul_readvariableop_resource:	–K
7lstm_117_lstm_cell_122_matmul_1_readvariableop_resource:
і–E
6lstm_117_lstm_cell_122_biasadd_readvariableop_resource:	–I
5lstm_118_lstm_cell_123_matmul_readvariableop_resource:
і–K
7lstm_118_lstm_cell_123_matmul_1_readvariableop_resource:
і–E
6lstm_118_lstm_cell_123_biasadd_readvariableop_resource:	–I
5lstm_119_lstm_cell_124_matmul_readvariableop_resource:
і–K
7lstm_119_lstm_cell_124_matmul_1_readvariableop_resource:
і–E
6lstm_119_lstm_cell_124_biasadd_readvariableop_resource:	–:
'dense_89_matmul_readvariableop_resource:	і6
(dense_89_biasadd_readvariableop_resource:
identityИҐdense_89/BiasAdd/ReadVariableOpҐdense_89/MatMul/ReadVariableOpҐ-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOpҐ,lstm_117/lstm_cell_122/MatMul/ReadVariableOpҐ.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOpҐlstm_117/whileҐ-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOpҐ,lstm_118/lstm_cell_123/MatMul/ReadVariableOpҐ.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOpҐlstm_118/whileҐ-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOpҐ,lstm_119/lstm_cell_124/MatMul/ReadVariableOpҐ.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOpҐlstm_119/whileD
lstm_117/ShapeShapeinputs*
T0*
_output_shapes
:f
lstm_117/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_117/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_117/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
lstm_117/strided_sliceStridedSlicelstm_117/Shape:output:0%lstm_117/strided_slice/stack:output:0'lstm_117/strided_slice/stack_1:output:0'lstm_117/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
lstm_117/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іО
lstm_117/zeros/packedPacklstm_117/strided_slice:output:0 lstm_117/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_117/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
lstm_117/zerosFilllstm_117/zeros/packed:output:0lstm_117/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€і\
lstm_117/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іТ
lstm_117/zeros_1/packedPacklstm_117/strided_slice:output:0"lstm_117/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_117/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    О
lstm_117/zeros_1Fill lstm_117/zeros_1/packed:output:0lstm_117/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іl
lstm_117/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_117/transpose	Transposeinputs lstm_117/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€V
lstm_117/Shape_1Shapelstm_117/transpose:y:0*
T0*
_output_shapes
:h
lstm_117/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_117/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_117/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
lstm_117/strided_slice_1StridedSlicelstm_117/Shape_1:output:0'lstm_117/strided_slice_1/stack:output:0)lstm_117/strided_slice_1/stack_1:output:0)lstm_117/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_117/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
lstm_117/TensorArrayV2TensorListReserve-lstm_117/TensorArrayV2/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“П
>lstm_117/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ы
0lstm_117/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_117/transpose:y:0Glstm_117/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“h
lstm_117/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_117/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_117/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
lstm_117/strided_slice_2StridedSlicelstm_117/transpose:y:0'lstm_117/strided_slice_2/stack:output:0)lstm_117/strided_slice_2/stack_1:output:0)lstm_117/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*
shrink_axis_mask£
,lstm_117/lstm_cell_122/MatMul/ReadVariableOpReadVariableOp5lstm_117_lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0≥
lstm_117/lstm_cell_122/MatMulMatMul!lstm_117/strided_slice_2:output:04lstm_117/lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–®
.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp7lstm_117_lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≠
lstm_117/lstm_cell_122/MatMul_1MatMullstm_117/zeros:output:06lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–™
lstm_117/lstm_cell_122/addAddV2'lstm_117/lstm_cell_122/MatMul:product:0)lstm_117/lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–°
-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp6lstm_117_lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0≥
lstm_117/lstm_cell_122/BiasAddBiasAddlstm_117/lstm_cell_122/add:z:05lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–h
&lstm_117/lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :€
lstm_117/lstm_cell_122/splitSplit/lstm_117/lstm_cell_122/split/split_dim:output:0'lstm_117/lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitГ
lstm_117/lstm_cell_122/SigmoidSigmoid%lstm_117/lstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_117/lstm_cell_122/Sigmoid_1Sigmoid%lstm_117/lstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іХ
lstm_117/lstm_cell_122/mulMul$lstm_117/lstm_cell_122/Sigmoid_1:y:0lstm_117/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€і}
lstm_117/lstm_cell_122/ReluRelu%lstm_117/lstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€і•
lstm_117/lstm_cell_122/mul_1Mul"lstm_117/lstm_cell_122/Sigmoid:y:0)lstm_117/lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЪ
lstm_117/lstm_cell_122/add_1AddV2lstm_117/lstm_cell_122/mul:z:0 lstm_117/lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_117/lstm_cell_122/Sigmoid_2Sigmoid%lstm_117/lstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_117/lstm_cell_122/Relu_1Relu lstm_117/lstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і©
lstm_117/lstm_cell_122/mul_2Mul$lstm_117/lstm_cell_122/Sigmoid_2:y:0+lstm_117/lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іw
&lstm_117/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ”
lstm_117/TensorArrayV2_1TensorListReserve/lstm_117/TensorArrayV2_1/element_shape:output:0!lstm_117/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“O
lstm_117/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_117/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
lstm_117/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Л
lstm_117/whileWhile$lstm_117/while/loop_counter:output:0*lstm_117/while/maximum_iterations:output:0lstm_117/time:output:0!lstm_117/TensorArrayV2_1:handle:0lstm_117/zeros:output:0lstm_117/zeros_1:output:0!lstm_117/strided_slice_1:output:0@lstm_117/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_117_lstm_cell_122_matmul_readvariableop_resource7lstm_117_lstm_cell_122_matmul_1_readvariableop_resource6lstm_117_lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_117_while_body_23347299*(
cond R
lstm_117_while_cond_23347298*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations К
9lstm_117/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ё
+lstm_117/TensorArrayV2Stack/TensorListStackTensorListStacklstm_117/while:output:3Blstm_117/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0q
lstm_117/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 lstm_117/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_117/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
lstm_117/strided_slice_3StridedSlice4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_117/strided_slice_3/stack:output:0)lstm_117/strided_slice_3/stack_1:output:0)lstm_117/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskn
lstm_117/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ≤
lstm_117/transpose_1	Transpose4lstm_117/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_117/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іd
lstm_117/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_118/ShapeShapelstm_117/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_118/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_118/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_118/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
lstm_118/strided_sliceStridedSlicelstm_118/Shape:output:0%lstm_118/strided_slice/stack:output:0'lstm_118/strided_slice/stack_1:output:0'lstm_118/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
lstm_118/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іО
lstm_118/zeros/packedPacklstm_118/strided_slice:output:0 lstm_118/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_118/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
lstm_118/zerosFilllstm_118/zeros/packed:output:0lstm_118/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€і\
lstm_118/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іТ
lstm_118/zeros_1/packedPacklstm_118/strided_slice:output:0"lstm_118/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_118/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    О
lstm_118/zeros_1Fill lstm_118/zeros_1/packed:output:0lstm_118/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іl
lstm_118/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Т
lstm_118/transpose	Transposelstm_117/transpose_1:y:0 lstm_118/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іV
lstm_118/Shape_1Shapelstm_118/transpose:y:0*
T0*
_output_shapes
:h
lstm_118/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_118/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_118/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
lstm_118/strided_slice_1StridedSlicelstm_118/Shape_1:output:0'lstm_118/strided_slice_1/stack:output:0)lstm_118/strided_slice_1/stack_1:output:0)lstm_118/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_118/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
lstm_118/TensorArrayV2TensorListReserve-lstm_118/TensorArrayV2/element_shape:output:0!lstm_118/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“П
>lstm_118/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ы
0lstm_118/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_118/transpose:y:0Glstm_118/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“h
lstm_118/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_118/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_118/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
lstm_118/strided_slice_2StridedSlicelstm_118/transpose:y:0'lstm_118/strided_slice_2/stack:output:0)lstm_118/strided_slice_2/stack_1:output:0)lstm_118/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_mask§
,lstm_118/lstm_cell_123/MatMul/ReadVariableOpReadVariableOp5lstm_118_lstm_cell_123_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≥
lstm_118/lstm_cell_123/MatMulMatMul!lstm_118/strided_slice_2:output:04lstm_118/lstm_cell_123/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–®
.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOpReadVariableOp7lstm_118_lstm_cell_123_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≠
lstm_118/lstm_cell_123/MatMul_1MatMullstm_118/zeros:output:06lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–™
lstm_118/lstm_cell_123/addAddV2'lstm_118/lstm_cell_123/MatMul:product:0)lstm_118/lstm_cell_123/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–°
-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOpReadVariableOp6lstm_118_lstm_cell_123_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0≥
lstm_118/lstm_cell_123/BiasAddBiasAddlstm_118/lstm_cell_123/add:z:05lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–h
&lstm_118/lstm_cell_123/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :€
lstm_118/lstm_cell_123/splitSplit/lstm_118/lstm_cell_123/split/split_dim:output:0'lstm_118/lstm_cell_123/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitГ
lstm_118/lstm_cell_123/SigmoidSigmoid%lstm_118/lstm_cell_123/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_118/lstm_cell_123/Sigmoid_1Sigmoid%lstm_118/lstm_cell_123/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іХ
lstm_118/lstm_cell_123/mulMul$lstm_118/lstm_cell_123/Sigmoid_1:y:0lstm_118/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€і}
lstm_118/lstm_cell_123/ReluRelu%lstm_118/lstm_cell_123/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€і•
lstm_118/lstm_cell_123/mul_1Mul"lstm_118/lstm_cell_123/Sigmoid:y:0)lstm_118/lstm_cell_123/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЪ
lstm_118/lstm_cell_123/add_1AddV2lstm_118/lstm_cell_123/mul:z:0 lstm_118/lstm_cell_123/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_118/lstm_cell_123/Sigmoid_2Sigmoid%lstm_118/lstm_cell_123/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_118/lstm_cell_123/Relu_1Relu lstm_118/lstm_cell_123/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і©
lstm_118/lstm_cell_123/mul_2Mul$lstm_118/lstm_cell_123/Sigmoid_2:y:0+lstm_118/lstm_cell_123/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іw
&lstm_118/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ”
lstm_118/TensorArrayV2_1TensorListReserve/lstm_118/TensorArrayV2_1/element_shape:output:0!lstm_118/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“O
lstm_118/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_118/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
lstm_118/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Л
lstm_118/whileWhile$lstm_118/while/loop_counter:output:0*lstm_118/while/maximum_iterations:output:0lstm_118/time:output:0!lstm_118/TensorArrayV2_1:handle:0lstm_118/zeros:output:0lstm_118/zeros_1:output:0!lstm_118/strided_slice_1:output:0@lstm_118/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_118_lstm_cell_123_matmul_readvariableop_resource7lstm_118_lstm_cell_123_matmul_1_readvariableop_resource6lstm_118_lstm_cell_123_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_118_while_body_23347438*(
cond R
lstm_118_while_cond_23347437*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations К
9lstm_118/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ё
+lstm_118/TensorArrayV2Stack/TensorListStackTensorListStacklstm_118/while:output:3Blstm_118/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0q
lstm_118/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 lstm_118/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_118/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
lstm_118/strided_slice_3StridedSlice4lstm_118/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_118/strided_slice_3/stack:output:0)lstm_118/strided_slice_3/stack_1:output:0)lstm_118/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskn
lstm_118/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ≤
lstm_118/transpose_1	Transpose4lstm_118/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_118/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іd
lstm_118/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    V
lstm_119/ShapeShapelstm_118/transpose_1:y:0*
T0*
_output_shapes
:f
lstm_119/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_119/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_119/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
lstm_119/strided_sliceStridedSlicelstm_119/Shape:output:0%lstm_119/strided_slice/stack:output:0'lstm_119/strided_slice/stack_1:output:0'lstm_119/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
lstm_119/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іО
lstm_119/zeros/packedPacklstm_119/strided_slice:output:0 lstm_119/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_119/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    И
lstm_119/zerosFilllstm_119/zeros/packed:output:0lstm_119/zeros/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€і\
lstm_119/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іТ
lstm_119/zeros_1/packedPacklstm_119/strided_slice:output:0"lstm_119/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:[
lstm_119/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    О
lstm_119/zeros_1Fill lstm_119/zeros_1/packed:output:0lstm_119/zeros_1/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іl
lstm_119/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Т
lstm_119/transpose	Transposelstm_118/transpose_1:y:0 lstm_119/transpose/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іV
lstm_119/Shape_1Shapelstm_119/transpose:y:0*
T0*
_output_shapes
:h
lstm_119/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_119/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_119/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
lstm_119/strided_slice_1StridedSlicelstm_119/Shape_1:output:0'lstm_119/strided_slice_1/stack:output:0)lstm_119/strided_slice_1/stack_1:output:0)lstm_119/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
$lstm_119/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
lstm_119/TensorArrayV2TensorListReserve-lstm_119/TensorArrayV2/element_shape:output:0!lstm_119/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“П
>lstm_119/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   ы
0lstm_119/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_119/transpose:y:0Glstm_119/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“h
lstm_119/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 lstm_119/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 lstm_119/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
lstm_119/strided_slice_2StridedSlicelstm_119/transpose:y:0'lstm_119/strided_slice_2/stack:output:0)lstm_119/strided_slice_2/stack_1:output:0)lstm_119/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_mask§
,lstm_119/lstm_cell_124/MatMul/ReadVariableOpReadVariableOp5lstm_119_lstm_cell_124_matmul_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≥
lstm_119/lstm_cell_124/MatMulMatMul!lstm_119/strided_slice_2:output:04lstm_119/lstm_cell_124/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–®
.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOpReadVariableOp7lstm_119_lstm_cell_124_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0≠
lstm_119/lstm_cell_124/MatMul_1MatMullstm_119/zeros:output:06lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–™
lstm_119/lstm_cell_124/addAddV2'lstm_119/lstm_cell_124/MatMul:product:0)lstm_119/lstm_cell_124/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–°
-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOpReadVariableOp6lstm_119_lstm_cell_124_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0≥
lstm_119/lstm_cell_124/BiasAddBiasAddlstm_119/lstm_cell_124/add:z:05lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–h
&lstm_119/lstm_cell_124/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :€
lstm_119/lstm_cell_124/splitSplit/lstm_119/lstm_cell_124/split/split_dim:output:0'lstm_119/lstm_cell_124/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitГ
lstm_119/lstm_cell_124/SigmoidSigmoid%lstm_119/lstm_cell_124/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_119/lstm_cell_124/Sigmoid_1Sigmoid%lstm_119/lstm_cell_124/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іХ
lstm_119/lstm_cell_124/mulMul$lstm_119/lstm_cell_124/Sigmoid_1:y:0lstm_119/zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€і}
lstm_119/lstm_cell_124/ReluRelu%lstm_119/lstm_cell_124/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€і•
lstm_119/lstm_cell_124/mul_1Mul"lstm_119/lstm_cell_124/Sigmoid:y:0)lstm_119/lstm_cell_124/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іЪ
lstm_119/lstm_cell_124/add_1AddV2lstm_119/lstm_cell_124/mul:z:0 lstm_119/lstm_cell_124/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іЕ
 lstm_119/lstm_cell_124/Sigmoid_2Sigmoid%lstm_119/lstm_cell_124/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_119/lstm_cell_124/Relu_1Relu lstm_119/lstm_cell_124/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€і©
lstm_119/lstm_cell_124/mul_2Mul$lstm_119/lstm_cell_124/Sigmoid_2:y:0+lstm_119/lstm_cell_124/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іw
&lstm_119/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   g
%lstm_119/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :а
lstm_119/TensorArrayV2_1TensorListReserve/lstm_119/TensorArrayV2_1/element_shape:output:0.lstm_119/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“O
lstm_119/timeConst*
_output_shapes
: *
dtype0*
value	B : l
!lstm_119/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€]
lstm_119/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Л
lstm_119/whileWhile$lstm_119/while/loop_counter:output:0*lstm_119/while/maximum_iterations:output:0lstm_119/time:output:0!lstm_119/TensorArrayV2_1:handle:0lstm_119/zeros:output:0lstm_119/zeros_1:output:0!lstm_119/strided_slice_1:output:0@lstm_119/TensorArrayUnstack/TensorListFromTensor:output_handle:05lstm_119_lstm_cell_124_matmul_readvariableop_resource7lstm_119_lstm_cell_124_matmul_1_readvariableop_resource6lstm_119_lstm_cell_124_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *(
body R
lstm_119_while_body_23347578*(
cond R
lstm_119_while_cond_23347577*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations К
9lstm_119/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   т
+lstm_119/TensorArrayV2Stack/TensorListStackTensorListStacklstm_119/while:output:3Blstm_119/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0*
num_elementsq
lstm_119/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€j
 lstm_119/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: j
 lstm_119/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
lstm_119/strided_slice_3StridedSlice4lstm_119/TensorArrayV2Stack/TensorListStack:tensor:0'lstm_119/strided_slice_3/stack:output:0)lstm_119/strided_slice_3/stack_1:output:0)lstm_119/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maskn
lstm_119/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ≤
lstm_119/transpose_1	Transpose4lstm_119/TensorArrayV2Stack/TensorListStack:tensor:0"lstm_119/transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€іd
lstm_119/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ]
dropout_72/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Ц
dropout_72/dropout/MulMul!lstm_119/strided_slice_3:output:0!dropout_72/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€іi
dropout_72/dropout/ShapeShape!lstm_119/strided_slice_3:output:0*
T0*
_output_shapes
:£
/dropout_72/dropout/random_uniform/RandomUniformRandomUniform!dropout_72/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€і*
dtype0f
!dropout_72/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>»
dropout_72/dropout/GreaterEqualGreaterEqual8dropout_72/dropout/random_uniform/RandomUniform:output:0*dropout_72/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€і_
dropout_72/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ј
dropout_72/dropout/SelectV2SelectV2#dropout_72/dropout/GreaterEqual:z:0dropout_72/dropout/Mul:z:0#dropout_72/dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іЗ
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes
:	і*
dtype0Щ
dense_89/MatMulMatMul$dropout_72/dropout/SelectV2:output:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_89/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€м
NoOpNoOp ^dense_89/BiasAdd/ReadVariableOp^dense_89/MatMul/ReadVariableOp.^lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp-^lstm_117/lstm_cell_122/MatMul/ReadVariableOp/^lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp^lstm_117/while.^lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp-^lstm_118/lstm_cell_123/MatMul/ReadVariableOp/^lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp^lstm_118/while.^lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp-^lstm_119/lstm_cell_124/MatMul/ReadVariableOp/^lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp^lstm_119/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 2B
dense_89/BiasAdd/ReadVariableOpdense_89/BiasAdd/ReadVariableOp2@
dense_89/MatMul/ReadVariableOpdense_89/MatMul/ReadVariableOp2^
-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp-lstm_117/lstm_cell_122/BiasAdd/ReadVariableOp2\
,lstm_117/lstm_cell_122/MatMul/ReadVariableOp,lstm_117/lstm_cell_122/MatMul/ReadVariableOp2`
.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp.lstm_117/lstm_cell_122/MatMul_1/ReadVariableOp2 
lstm_117/whilelstm_117/while2^
-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp-lstm_118/lstm_cell_123/BiasAdd/ReadVariableOp2\
,lstm_118/lstm_cell_123/MatMul/ReadVariableOp,lstm_118/lstm_cell_123/MatMul/ReadVariableOp2`
.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp.lstm_118/lstm_cell_123/MatMul_1/ReadVariableOp2 
lstm_118/whilelstm_118/while2^
-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp-lstm_119/lstm_cell_124/BiasAdd/ReadVariableOp2\
,lstm_119/lstm_cell_124/MatMul/ReadVariableOp,lstm_119/lstm_cell_124/MatMul/ReadVariableOp2`
.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp.lstm_119/lstm_cell_124/MatMul_1/ReadVariableOp2 
lstm_119/whilelstm_119/while:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ

£
&__inference_signature_wrapper_23346756
lstm_117_input
unknown:	–
	unknown_0:
і–
	unknown_1:	–
	unknown_2:
і–
	unknown_3:
і–
	unknown_4:	–
	unknown_5:
і–
	unknown_6:
і–
	unknown_7:	–
	unknown_8:	і
	unknown_9:
identityИҐStatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCalllstm_117_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_23344459o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:€€€€€€€€€: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€
(
_user_specified_namelstm_117_input
АK
•
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348150

inputs?
,lstm_cell_122_matmul_readvariableop_resource:	–B
.lstm_cell_122_matmul_1_readvariableop_resource:
і–<
-lstm_cell_122_biasadd_readvariableop_resource:	–
identityИҐ$lstm_cell_122/BiasAdd/ReadVariableOpҐ#lstm_cell_122/MatMul/ReadVariableOpҐ%lstm_cell_122/MatMul_1/ReadVariableOpҐwhile;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іs
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
:€€€€€€€€€іS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :іw
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
:€€€€€€€€€іc
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
shrink_axis_maskС
#lstm_cell_122/MatMul/ReadVariableOpReadVariableOp,lstm_cell_122_matmul_readvariableop_resource*
_output_shapes
:	–*
dtype0Ш
lstm_cell_122/MatMulMatMulstrided_slice_2:output:0+lstm_cell_122/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–Ц
%lstm_cell_122/MatMul_1/ReadVariableOpReadVariableOp.lstm_cell_122_matmul_1_readvariableop_resource* 
_output_shapes
:
і–*
dtype0Т
lstm_cell_122/MatMul_1MatMulzeros:output:0-lstm_cell_122/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–П
lstm_cell_122/addAddV2lstm_cell_122/MatMul:product:0 lstm_cell_122/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€–П
$lstm_cell_122/BiasAdd/ReadVariableOpReadVariableOp-lstm_cell_122_biasadd_readvariableop_resource*
_output_shapes	
:–*
dtype0Ш
lstm_cell_122/BiasAddBiasAddlstm_cell_122/add:z:0,lstm_cell_122/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€–_
lstm_cell_122/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :д
lstm_cell_122/splitSplit&lstm_cell_122/split/split_dim:output:0lstm_cell_122/BiasAdd:output:0*
T0*d
_output_shapesR
P:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і:€€€€€€€€€і*
	num_splitq
lstm_cell_122/SigmoidSigmoidlstm_cell_122/split:output:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_1Sigmoidlstm_cell_122/split:output:1*
T0*(
_output_shapes
:€€€€€€€€€іz
lstm_cell_122/mulMullstm_cell_122/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€іk
lstm_cell_122/ReluRelulstm_cell_122/split:output:2*
T0*(
_output_shapes
:€€€€€€€€€іК
lstm_cell_122/mul_1Mullstm_cell_122/Sigmoid:y:0 lstm_cell_122/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€і
lstm_cell_122/add_1AddV2lstm_cell_122/mul:z:0lstm_cell_122/mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іs
lstm_cell_122/Sigmoid_2Sigmoidlstm_cell_122/split:output:3*
T0*(
_output_shapes
:€€€€€€€€€іh
lstm_cell_122/Relu_1Relulstm_cell_122/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€іО
lstm_cell_122/mul_2Mullstm_cell_122/Sigmoid_2:y:0"lstm_cell_122/Relu_1:activations:0*
T0*(
_output_shapes
:€€€€€€€€€іn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   Є
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
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
value	B : Н
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_cell_122_matmul_readvariableop_resource.lstm_cell_122_matmul_1_readvariableop_resource-lstm_cell_122_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_23348066*
condR
while_cond_23348065*M
output_shapes<
:: : : : :€€€€€€€€€і:€€€€€€€€€і: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€і   √
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:€€€€€€€€€і*
element_dtype0h
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
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:€€€€€€€€€і*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:€€€€€€€€€і[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€і√
NoOpNoOp%^lstm_cell_122/BiasAdd/ReadVariableOp$^lstm_cell_122/MatMul/ReadVariableOp&^lstm_cell_122/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€: : : 2L
$lstm_cell_122/BiasAdd/ReadVariableOp$lstm_cell_122/BiasAdd/ReadVariableOp2J
#lstm_cell_122/MatMul/ReadVariableOp#lstm_cell_122/MatMul/ReadVariableOp2N
%lstm_cell_122/MatMul_1/ReadVariableOp%lstm_cell_122/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*љ
serving_default©
M
lstm_117_input;
 serving_default_lstm_117_input:0€€€€€€€€€<
dense_890
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Дж
П
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
Џ
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
Џ
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
Џ
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
Љ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator"
_tf_keras_layer
ї
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
 
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
х
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32К
0__inference_sequential_91_layer_call_fn_23346021
0__inference_sequential_91_layer_call_fn_23346783
0__inference_sequential_91_layer_call_fn_23346810
0__inference_sequential_91_layer_call_fn_23346663њ
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
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
б
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32ц
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347240
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347677
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346694
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346725њ
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
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
’B“
#__inference__wrapped_model_23344459lstm_117_input"Ш
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
є

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
ц
]trace_0
^trace_1
_trace_2
`trace_32Л
+__inference_lstm_117_layer_call_fn_23347688
+__inference_lstm_117_layer_call_fn_23347699
+__inference_lstm_117_layer_call_fn_23347710
+__inference_lstm_117_layer_call_fn_23347721‘
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
 z]trace_0z^trace_1z_trace_2z`trace_3
в
atrace_0
btrace_1
ctrace_2
dtrace_32ч
F__inference_lstm_117_layer_call_and_return_conditional_losses_23347864
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348007
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348150
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348293‘
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
 zatrace_0zbtrace_1zctrace_2zdtrace_3
"
_generic_user_object
ш
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
є

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
ц
strace_0
ttrace_1
utrace_2
vtrace_32Л
+__inference_lstm_118_layer_call_fn_23348304
+__inference_lstm_118_layer_call_fn_23348315
+__inference_lstm_118_layer_call_fn_23348326
+__inference_lstm_118_layer_call_fn_23348337‘
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
 zstrace_0zttrace_1zutrace_2zvtrace_3
в
wtrace_0
xtrace_1
ytrace_2
ztrace_32ч
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348480
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348623
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348766
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348909‘
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
 zwtrace_0zxtrace_1zytrace_2zztrace_3
"
_generic_user_object
ы
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+А&call_and_return_all_conditional_losses
Б_random_generator
В
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
њ
Гstates
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ю
Йtrace_0
Кtrace_1
Лtrace_2
Мtrace_32Л
+__inference_lstm_119_layer_call_fn_23348920
+__inference_lstm_119_layer_call_fn_23348931
+__inference_lstm_119_layer_call_fn_23348942
+__inference_lstm_119_layer_call_fn_23348953‘
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
 zЙtrace_0zКtrace_1zЛtrace_2zМtrace_3
к
Нtrace_0
Оtrace_1
Пtrace_2
Рtrace_32ч
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349098
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349243
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349388
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349533‘
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
 zНtrace_0zОtrace_1zПtrace_2zРtrace_3
"
_generic_user_object
А
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator
Ш
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
≤
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ѕ
Юtrace_0
Яtrace_12Ф
-__inference_dropout_72_layer_call_fn_23349538
-__inference_dropout_72_layer_call_fn_23349543≥
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
 zЮtrace_0zЯtrace_1
Е
†trace_0
°trace_12 
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349548
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349560≥
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
 z†trace_0z°trace_1
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
≤
Ґnon_trainable_variables
£layers
§metrics
 •layer_regularization_losses
¶layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
с
Іtrace_02“
+__inference_dense_89_layer_call_fn_23349569Ґ
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
 zІtrace_0
М
®trace_02н
F__inference_dense_89_layer_call_and_return_conditional_losses_23349579Ґ
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
 z®trace_0
": 	і2dense_89/kernel
:2dense_89/bias
0:.	–2lstm_117/lstm_cell_122/kernel
;:9
і–2'lstm_117/lstm_cell_122/recurrent_kernel
*:(–2lstm_117/lstm_cell_122/bias
1:/
і–2lstm_118/lstm_cell_123/kernel
;:9
і–2'lstm_118/lstm_cell_123/recurrent_kernel
*:(–2lstm_118/lstm_cell_123/bias
1:/
і–2lstm_119/lstm_cell_124/kernel
;:9
і–2'lstm_119/lstm_cell_124/recurrent_kernel
*:(–2lstm_119/lstm_cell_124/bias
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
©0
™1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЙBЖ
0__inference_sequential_91_layer_call_fn_23346021lstm_117_input"њ
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
0__inference_sequential_91_layer_call_fn_23346783inputs"њ
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
0__inference_sequential_91_layer_call_fn_23346810inputs"њ
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
ЙBЖ
0__inference_sequential_91_layer_call_fn_23346663lstm_117_input"њ
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
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347240inputs"њ
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
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347677inputs"њ
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
§B°
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346694lstm_117_input"њ
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
§B°
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346725lstm_117_input"њ
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
д
P0
Ђ1
ђ2
≠3
Ѓ4
ѓ5
∞6
±7
≤8
≥9
і10
µ11
ґ12
Ј13
Є14
є15
Ї16
ї17
Љ18
љ19
Њ20
њ21
ј22"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
y
Ђ0
≠1
ѓ2
±3
≥4
µ5
Ј6
є7
ї8
љ9
њ10"
trackable_list_wrapper
y
ђ0
Ѓ1
∞2
≤3
і4
ґ5
Є6
Ї7
Љ8
Њ9
ј10"
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
‘B—
&__inference_signature_wrapper_23346756lstm_117_input"Ф
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
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
УBР
+__inference_lstm_117_layer_call_fn_23347688inputs_0"‘
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
УBР
+__inference_lstm_117_layer_call_fn_23347699inputs_0"‘
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
СBО
+__inference_lstm_117_layer_call_fn_23347710inputs"‘
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
СBО
+__inference_lstm_117_layer_call_fn_23347721inputs"‘
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
ЃBЂ
F__inference_lstm_117_layer_call_and_return_conditional_losses_23347864inputs_0"‘
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
ЃBЂ
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348007inputs_0"‘
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
ђB©
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348150inputs"‘
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
ђB©
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348293inputs"‘
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
≤
Ѕnon_trainable_variables
¬layers
√metrics
 ƒlayer_regularization_losses
≈layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
я
∆trace_0
«trace_12§
0__inference_lstm_cell_122_layer_call_fn_23349596
0__inference_lstm_cell_122_layer_call_fn_23349613љ
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
 z∆trace_0z«trace_1
Х
»trace_0
…trace_12Џ
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349645
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349677љ
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
 z»trace_0z…trace_1
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
УBР
+__inference_lstm_118_layer_call_fn_23348304inputs_0"‘
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
УBР
+__inference_lstm_118_layer_call_fn_23348315inputs_0"‘
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
СBО
+__inference_lstm_118_layer_call_fn_23348326inputs"‘
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
СBО
+__inference_lstm_118_layer_call_fn_23348337inputs"‘
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
ЃBЂ
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348480inputs_0"‘
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
ЃBЂ
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348623inputs_0"‘
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
ђB©
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348766inputs"‘
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
ђB©
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348909inputs"‘
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
і
 non_trainable_variables
Ћlayers
ћmetrics
 Ќlayer_regularization_losses
ќlayer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
я
ѕtrace_0
–trace_12§
0__inference_lstm_cell_123_layer_call_fn_23349694
0__inference_lstm_cell_123_layer_call_fn_23349711љ
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
 zѕtrace_0z–trace_1
Х
—trace_0
“trace_12Џ
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349743
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349775љ
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
 z—trace_0z“trace_1
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
УBР
+__inference_lstm_119_layer_call_fn_23348920inputs_0"‘
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
УBР
+__inference_lstm_119_layer_call_fn_23348931inputs_0"‘
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
СBО
+__inference_lstm_119_layer_call_fn_23348942inputs"‘
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
СBО
+__inference_lstm_119_layer_call_fn_23348953inputs"‘
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
ЃBЂ
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349098inputs_0"‘
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
ЃBЂ
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349243inputs_0"‘
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
ђB©
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349388inputs"‘
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
ђB©
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349533inputs"‘
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
Є
”non_trainable_variables
‘layers
’metrics
 ÷layer_regularization_losses
„layer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
я
Ўtrace_0
ўtrace_12§
0__inference_lstm_cell_124_layer_call_fn_23349792
0__inference_lstm_cell_124_layer_call_fn_23349809љ
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
 zЎtrace_0zўtrace_1
Х
Џtrace_0
џtrace_12Џ
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349841
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349873љ
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
 zЏtrace_0zџtrace_1
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
-__inference_dropout_72_layer_call_fn_23349538inputs"≥
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
-__inference_dropout_72_layer_call_fn_23349543inputs"≥
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
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349548inputs"≥
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
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349560inputs"≥
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
+__inference_dense_89_layer_call_fn_23349569inputs"Ґ
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
F__inference_dense_89_layer_call_and_return_conditional_losses_23349579inputs"Ґ
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
R
№	variables
Ё	keras_api

ёtotal

яcount"
_tf_keras_metric
c
а	variables
б	keras_api

вtotal

гcount
д
_fn_kwargs"
_tf_keras_metric
5:3	–2$Adam/m/lstm_117/lstm_cell_122/kernel
5:3	–2$Adam/v/lstm_117/lstm_cell_122/kernel
@:>
і–2.Adam/m/lstm_117/lstm_cell_122/recurrent_kernel
@:>
і–2.Adam/v/lstm_117/lstm_cell_122/recurrent_kernel
/:-–2"Adam/m/lstm_117/lstm_cell_122/bias
/:-–2"Adam/v/lstm_117/lstm_cell_122/bias
6:4
і–2$Adam/m/lstm_118/lstm_cell_123/kernel
6:4
і–2$Adam/v/lstm_118/lstm_cell_123/kernel
@:>
і–2.Adam/m/lstm_118/lstm_cell_123/recurrent_kernel
@:>
і–2.Adam/v/lstm_118/lstm_cell_123/recurrent_kernel
/:-–2"Adam/m/lstm_118/lstm_cell_123/bias
/:-–2"Adam/v/lstm_118/lstm_cell_123/bias
6:4
і–2$Adam/m/lstm_119/lstm_cell_124/kernel
6:4
і–2$Adam/v/lstm_119/lstm_cell_124/kernel
@:>
і–2.Adam/m/lstm_119/lstm_cell_124/recurrent_kernel
@:>
і–2.Adam/v/lstm_119/lstm_cell_124/recurrent_kernel
/:-–2"Adam/m/lstm_119/lstm_cell_124/bias
/:-–2"Adam/v/lstm_119/lstm_cell_124/bias
':%	і2Adam/m/dense_89/kernel
':%	і2Adam/v/dense_89/kernel
 :2Adam/m/dense_89/bias
 :2Adam/v/dense_89/bias
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
УBР
0__inference_lstm_cell_122_layer_call_fn_23349596inputsstates_0states_1"љ
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
УBР
0__inference_lstm_cell_122_layer_call_fn_23349613inputsstates_0states_1"љ
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
ЃBЂ
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349645inputsstates_0states_1"љ
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
ЃBЂ
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349677inputsstates_0states_1"љ
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
УBР
0__inference_lstm_cell_123_layer_call_fn_23349694inputsstates_0states_1"љ
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
УBР
0__inference_lstm_cell_123_layer_call_fn_23349711inputsstates_0states_1"љ
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
ЃBЂ
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349743inputsstates_0states_1"љ
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
ЃBЂ
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349775inputsstates_0states_1"љ
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
УBР
0__inference_lstm_cell_124_layer_call_fn_23349792inputsstates_0states_1"љ
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
УBР
0__inference_lstm_cell_124_layer_call_fn_23349809inputsstates_0states_1"љ
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
ЃBЂ
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349841inputsstates_0states_1"љ
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
ЃBЂ
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349873inputsstates_0states_1"љ
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
0
ё0
я1"
trackable_list_wrapper
.
№	variables"
_generic_user_object
:  (2total
:  (2count
0
в0
г1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper¶
#__inference__wrapped_model_233444599:;<=>?@A78;Ґ8
1Ґ.
,К)
lstm_117_input€€€€€€€€€
™ "3™0
.
dense_89"К
dense_89€€€€€€€€€Ѓ
F__inference_dense_89_layer_call_and_return_conditional_losses_23349579d780Ґ-
&Ґ#
!К
inputs€€€€€€€€€і
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
+__inference_dense_89_layer_call_fn_23349569Y780Ґ-
&Ґ#
!К
inputs€€€€€€€€€і
™ "!К
unknown€€€€€€€€€±
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349548e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€і
p 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€і
Ъ ±
H__inference_dropout_72_layer_call_and_return_conditional_losses_23349560e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€і
p
™ "-Ґ*
#К 
tensor_0€€€€€€€€€і
Ъ Л
-__inference_dropout_72_layer_call_fn_23349538Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€і
p 
™ ""К
unknown€€€€€€€€€іЛ
-__inference_dropout_72_layer_call_fn_23349543Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€і
p
™ ""К
unknown€€€€€€€€€іЁ
F__inference_lstm_117_layer_call_and_return_conditional_losses_23347864Т9:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ ":Ґ7
0К-
tensor_0€€€€€€€€€€€€€€€€€€і
Ъ Ё
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348007Т9:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ ":Ґ7
0К-
tensor_0€€€€€€€€€€€€€€€€€€і
Ъ √
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348150y9:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "1Ґ.
'К$
tensor_0€€€€€€€€€і
Ъ √
F__inference_lstm_117_layer_call_and_return_conditional_losses_23348293y9:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "1Ґ.
'К$
tensor_0€€€€€€€€€і
Ъ Ј
+__inference_lstm_117_layer_call_fn_23347688З9:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p 

 
™ "/К,
unknown€€€€€€€€€€€€€€€€€€іЈ
+__inference_lstm_117_layer_call_fn_23347699З9:;OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€

 
p

 
™ "/К,
unknown€€€€€€€€€€€€€€€€€€іЭ
+__inference_lstm_117_layer_call_fn_23347710n9:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p 

 
™ "&К#
unknown€€€€€€€€€іЭ
+__inference_lstm_117_layer_call_fn_23347721n9:;?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€

 
p

 
™ "&К#
unknown€€€€€€€€€іё
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348480У<=>PҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p 

 
™ ":Ґ7
0К-
tensor_0€€€€€€€€€€€€€€€€€€і
Ъ ё
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348623У<=>PҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p

 
™ ":Ґ7
0К-
tensor_0€€€€€€€€€€€€€€€€€€і
Ъ ƒ
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348766z<=>@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p 

 
™ "1Ґ.
'К$
tensor_0€€€€€€€€€і
Ъ ƒ
F__inference_lstm_118_layer_call_and_return_conditional_losses_23348909z<=>@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p

 
™ "1Ґ.
'К$
tensor_0€€€€€€€€€і
Ъ Є
+__inference_lstm_118_layer_call_fn_23348304И<=>PҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p 

 
™ "/К,
unknown€€€€€€€€€€€€€€€€€€іЄ
+__inference_lstm_118_layer_call_fn_23348315И<=>PҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p

 
™ "/К,
unknown€€€€€€€€€€€€€€€€€€іЮ
+__inference_lstm_118_layer_call_fn_23348326o<=>@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p 

 
™ "&К#
unknown€€€€€€€€€іЮ
+__inference_lstm_118_layer_call_fn_23348337o<=>@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p

 
™ "&К#
unknown€€€€€€€€€і—
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349098Ж?@APҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p 

 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€і
Ъ —
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349243Ж?@APҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p

 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€і
Ъ ј
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349388v?@A@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p 

 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€і
Ъ ј
F__inference_lstm_119_layer_call_and_return_conditional_losses_23349533v?@A@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p

 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€і
Ъ ™
+__inference_lstm_119_layer_call_fn_23348920{?@APҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p 

 
™ ""К
unknown€€€€€€€€€і™
+__inference_lstm_119_layer_call_fn_23348931{?@APҐM
FҐC
5Ъ2
0К-
inputs_0€€€€€€€€€€€€€€€€€€і

 
p

 
™ ""К
unknown€€€€€€€€€іЪ
+__inference_lstm_119_layer_call_fn_23348942k?@A@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p 

 
™ ""К
unknown€€€€€€€€€іЪ
+__inference_lstm_119_layer_call_fn_23348953k?@A@Ґ=
6Ґ3
%К"
inputs€€€€€€€€€і

 
p

 
™ ""К
unknown€€€€€€€€€ік
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349645Ъ9:;ВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p 
™ "НҐЙ
БҐ~
%К"

tensor_0_0€€€€€€€€€і
UЪR
'К$
tensor_0_1_0€€€€€€€€€і
'К$
tensor_0_1_1€€€€€€€€€і
Ъ к
K__inference_lstm_cell_122_layer_call_and_return_conditional_losses_23349677Ъ9:;ВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p
™ "НҐЙ
БҐ~
%К"

tensor_0_0€€€€€€€€€і
UЪR
'К$
tensor_0_1_0€€€€€€€€€і
'К$
tensor_0_1_1€€€€€€€€€і
Ъ Љ
0__inference_lstm_cell_122_layer_call_fn_23349596З9:;ВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p 
™ "{Ґx
#К 
tensor_0€€€€€€€€€і
QЪN
%К"

tensor_1_0€€€€€€€€€і
%К"

tensor_1_1€€€€€€€€€іЉ
0__inference_lstm_cell_122_layer_call_fn_23349613З9:;ВҐ
xҐu
 К
inputs€€€€€€€€€
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p
™ "{Ґx
#К 
tensor_0€€€€€€€€€і
QЪN
%К"

tensor_1_0€€€€€€€€€і
%К"

tensor_1_1€€€€€€€€€ім
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349743Ь<=>ДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p 
™ "НҐЙ
БҐ~
%К"

tensor_0_0€€€€€€€€€і
UЪR
'К$
tensor_0_1_0€€€€€€€€€і
'К$
tensor_0_1_1€€€€€€€€€і
Ъ м
K__inference_lstm_cell_123_layer_call_and_return_conditional_losses_23349775Ь<=>ДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p
™ "НҐЙ
БҐ~
%К"

tensor_0_0€€€€€€€€€і
UЪR
'К$
tensor_0_1_0€€€€€€€€€і
'К$
tensor_0_1_1€€€€€€€€€і
Ъ Њ
0__inference_lstm_cell_123_layer_call_fn_23349694Й<=>ДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p 
™ "{Ґx
#К 
tensor_0€€€€€€€€€і
QЪN
%К"

tensor_1_0€€€€€€€€€і
%К"

tensor_1_1€€€€€€€€€іЊ
0__inference_lstm_cell_123_layer_call_fn_23349711Й<=>ДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p
™ "{Ґx
#К 
tensor_0€€€€€€€€€і
QЪN
%К"

tensor_1_0€€€€€€€€€і
%К"

tensor_1_1€€€€€€€€€ім
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349841Ь?@AДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p 
™ "НҐЙ
БҐ~
%К"

tensor_0_0€€€€€€€€€і
UЪR
'К$
tensor_0_1_0€€€€€€€€€і
'К$
tensor_0_1_1€€€€€€€€€і
Ъ м
K__inference_lstm_cell_124_layer_call_and_return_conditional_losses_23349873Ь?@AДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p
™ "НҐЙ
БҐ~
%К"

tensor_0_0€€€€€€€€€і
UЪR
'К$
tensor_0_1_0€€€€€€€€€і
'К$
tensor_0_1_1€€€€€€€€€і
Ъ Њ
0__inference_lstm_cell_124_layer_call_fn_23349792Й?@AДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p 
™ "{Ґx
#К 
tensor_0€€€€€€€€€і
QЪN
%К"

tensor_1_0€€€€€€€€€і
%К"

tensor_1_1€€€€€€€€€іЊ
0__inference_lstm_cell_124_layer_call_fn_23349809Й?@AДҐА
yҐv
!К
inputs€€€€€€€€€і
MҐJ
#К 
states_0€€€€€€€€€і
#К 
states_1€€€€€€€€€і
p
™ "{Ґx
#К 
tensor_0€€€€€€€€€і
QЪN
%К"

tensor_1_0€€€€€€€€€і
%К"

tensor_1_1€€€€€€€€€і–
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346694А9:;<=>?@A78CҐ@
9Ґ6
,К)
lstm_117_input€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ –
K__inference_sequential_91_layer_call_and_return_conditional_losses_23346725А9:;<=>?@A78CҐ@
9Ґ6
,К)
lstm_117_input€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ «
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347240x9:;<=>?@A78;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ «
K__inference_sequential_91_layer_call_and_return_conditional_losses_23347677x9:;<=>?@A78;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ©
0__inference_sequential_91_layer_call_fn_23346021u9:;<=>?@A78CҐ@
9Ґ6
,К)
lstm_117_input€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€©
0__inference_sequential_91_layer_call_fn_23346663u9:;<=>?@A78CҐ@
9Ґ6
,К)
lstm_117_input€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€°
0__inference_sequential_91_layer_call_fn_23346783m9:;<=>?@A78;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€°
0__inference_sequential_91_layer_call_fn_23346810m9:;<=>?@A78;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€Љ
&__inference_signature_wrapper_23346756С9:;<=>?@A78MҐJ
Ґ 
C™@
>
lstm_117_input,К)
lstm_117_input€€€€€€€€€"3™0
.
dense_89"К
dense_89€€€€€€€€€