open Base
open Torch

type t = Tensor.t
[@@deriving equal]

(** Constants *)

val float_eps : float
val float_tiny : float

(** Simplified Kinds *)

type kind =
  | Bool
  | Int
  | Float
  | Complex
[@@deriving equal]

val string_of_kind : kind -> string
val to_kind : t -> kind:kind -> t
val kind : t -> kind

(** Generated APIs *)

val _sample_dirichlet : t -> t
val _standard_gamma : t -> t

val abs : t -> t
val acos : t -> t
val acosh : t -> t
val add : t -> t -> t
val add_scalar : t -> 'a Scalar.t -> t
val all : t -> t
val all_dim : dim:int -> ?keepdim:bool -> t -> t
val alpha_dropout : t -> p:float -> train:bool -> t
val amax : dim:int list -> ?keepdim:bool -> t -> t
val amin : dim:int list -> ?keepdim:bool -> t -> t
val angle : t -> t
val any : t -> t
val any_dim : dim:int -> ?keepdim:bool -> t -> t
val arange : ?kind:kind -> ?device:Device.t -> 'a Scalar.t -> t
val arange_start : ?kind:kind -> ?device:Device.t -> start:'a Scalar.t -> 'a Scalar.t -> t
val arange_start_step : ?kind:kind -> ?device:Device.t -> start:'a Scalar.t -> step:'a Scalar.t -> 'a Scalar.t -> t
val argmax : dim:int -> ?keepdim:bool -> t -> t
val argmin : dim:int -> ?keepdim:bool -> t -> t
val argsort : ?dim:int -> ?descending:bool -> t -> t
val as_strided : size:int list -> stride:int list -> ?storage_offset:int -> t -> t
val asin : t -> t
val asinh : t -> t
val atan : t -> t
val atan2 : t -> t -> t
val atanh : t -> t

val bernoulli : t -> t
val bernoulli_float : ?device:Device.t -> p:float -> int list -> t
val bilinear : weight:t -> ?bias:t -> input1:t -> input2:t -> t
val binary_cross_entropy : target:t -> ?weight:t -> ?reduction:Torch_core.Reduction.t -> t -> t

(** [binary_cross_entropy_with_logits ~target ?(reduction='mean') logits] measures Binary Cross Entropy
    between target and logits. *)
val binary_cross_entropy_with_logits :
  target:t
  -> ?weight:t
  -> ?pos_weight:t
  -> ?reduction:Torch_core.Reduction.t
  -> t
  -> t

val bincount : ?weights:t -> ?minlength:int -> t -> t
val binomial : count:t -> prob:t -> t
val bitwise_and : t -> 'a Scalar.t -> t
val bitwise_and_tensor : t -> t -> t
val bitwise_not : t -> t
val bitwise_or : t -> 'a Scalar.t -> t
val bitwise_or_tensor : t -> t -> t
val bitwise_xor : t -> 'a Scalar.t -> t
val bitwise_xor_tensor : t -> t -> t
val block_diag : t list -> t
val broadcast_tensors : t list -> t list
val broadcast_to : t -> size:int list -> t

val cartesian_prod : t list -> t

(** [cat ?(dim=0) ts] concatenates the given sequence of tensors [ts] in the given dimension [dim]. *)
val cat : ?dim:int -> t list -> t

val cauchy : ?device:Device.t -> median:float -> sigma:float -> int list -> t
val ceil : t -> t
val celu : t -> t
val chain_matmul : matrices:t list -> t

(** [cholesky ?(upper=false) t] computes the Cholesky decomposition of a symmetric positive-definite
    matrix or for batches of symmetric positive-definite matrices.  *)
val cholesky : ?upper:bool -> t -> t

val cholesky_inverse : ?upper:bool -> t -> t
val cholesky_solve : ?upper:bool -> input2:t -> t -> t
val chunk : chunks:int -> ?dim:int -> t -> t list

(** [clamp t ~min ~max] clamp all elements in [t] into the range [[min, max]].  *)
val clamp : t -> min:'a Scalar.t -> max:'a Scalar.t -> t

val clamp_max : t -> max:'a Scalar.t -> t
val clamp_max_tensor : t -> max:t -> t
val clamp_min : t -> min:'a Scalar.t -> t
val clamp_min_tensor : t -> min:t -> t
val clamp_tensor : ?min:t -> ?max:t -> t -> t
val clone : t -> t
val column_stack : t list -> t
val combinations : ?r:int -> ?with_replacement:bool -> t -> t
val complex : real:t -> imag:t -> t
val conj : t -> t
val corrcoef : t -> t
val cos : t -> t
val cosh : t -> t
val cosine_similarity : ?dim:int -> ?eps:float -> t -> t -> t
val cross_entropy_loss : target:t -> ?weight:t -> ?reduction:Torch_core.Reduction.t -> ?ignore_index:int -> ?label_smoothing:float -> t -> t
val cummax : t -> dim:int -> t * t
val cummin : t -> dim:int -> t * t

(** [cumprod ~dim t] returns the cumulative product of elements of [t] in the dimension [dim]. *)
val cumprod : dim:int -> ?kind:kind -> t -> t

val cumsum : dim:int -> ?kind:kind -> t -> t

val data : t -> t
val deg2rad : t -> t
val det : t -> t

(** [detach t] returns a new Tensor, detached from the current graph. The result will never require gradient. *)
val detach : t -> t

val diag : ?diagonal:int -> t -> t

(** [diag_embed ?(offset=0) ?(dim1=-2) ?(dim2=-1) t] creates a tensor whose diagonals of certain 2D planes
    (specified by [dim1] and [dim2]) are filled by [t]. *)
val diag_embed : ?offset:int -> ?dim1:int -> ?dim2:int -> t -> t

val diagflat : ?offset:int -> t -> t

(** [diagonal ?(offset=0) ?(dim1=0) ?(dim2=1) t] returns a partial view of [t] with the its diagonal elements
    with respect to [dim1] and [dim2] appended as a dimension at the end of the shape. *)
val diagonal : ?offset:int -> ?dim1:int -> ?dim2:int -> t -> t

val diff : ?n:int -> ?dim:int -> ?prepend:t -> ?append:t -> t -> t
val digamma : t -> t
val dist : t -> t -> t
val div : t -> t -> t
val div_scalar : t -> 'a Scalar.t -> t
val dot : t -> t -> t
val dropout : ?p:float -> ?train:bool -> t -> t
val dsplit : t -> sections:int -> t list
val dsplit_array : t -> indices:int list -> t list
val dstack : t list -> t

val eig : ?eigenvectors:bool -> t -> t * t
val einsum : equation:string -> t list -> t
val elu : t -> t
val empty : ?kind:kind -> ?device:Device.t -> size:int list -> t
val empty_like : t -> t
val eq : t -> 'a Scalar.t -> t
val eq_tensor : t -> t -> t
val erf : t -> t
val erfc : t -> t
val erfinv : t -> t
val exp : t -> t
val exp2 : t -> t
val expand : size:int list -> ?implicit:bool -> t -> t
val expand_as : t -> t -> t
val expm1 : t -> t
val eye : ?kind:kind -> ?device:Device.t -> int -> t
val eye_m : ?kind:kind -> ?device:Device.t -> n:int -> m:int -> t

val fill_ : t -> value:'a Scalar.t -> t
val fill_diagonal_ : ?wrap:bool -> fill_value:'a Scalar.t -> t -> t
val fill_tensor_ : t -> value:t -> t
val flatten : ?start_dim:int -> ?end_dim:int -> t -> t
val flip : t -> dims:int list -> t
val fliplr : t -> t
val flipud : t -> t
val float_power : t -> exponent:t -> t
val float_power_scalar : 'a Scalar.t -> exponent:t -> t
val float_power_tensor_scalar : t -> exponent:'a Scalar.t -> t
val floor : t -> t
val floor_divide : t -> t -> t
val floor_divide_scalar : t -> 'a Scalar.t -> t
val fmax : t -> t -> t
val fmin : t -> t -> t
val fmod : t -> 'a Scalar.t -> t
val fmod_tensor : t -> t -> t
val frac : t -> t
val frexp : t -> t * t
val frobenius_norm : t -> t
val full : ?kind:kind -> ?device:Device.t -> fill_value:'a Scalar.t -> int list -> t
val full_like : t -> fill_value:'a Scalar.t -> t

(** [gather ~dim ~index ?(sparse_grad=false) t] gathers values along an axis specified by [dim]. *)
val gather : dim:int -> index:t -> ?sparse_grad:bool -> t -> t

val gcd : t -> t -> t
val ge : t -> 'a Scalar.t -> t
val ge_tensor : t -> t -> t
val gelu : t -> t
val geometric : ?device:Device.t -> p:float -> int list -> t
val grad : t -> t
val gt : t -> 'a Scalar.t -> t
val gt_tensor : t -> t -> t

val hinge_embedding_loss : target:t -> ?margin:float -> ?reduction:Torch_core.Reduction.t -> t -> t
val histc : t -> bins:int -> t
val hsplit : t -> sections:int -> t list
val hsplit_array : t -> indices:int list -> t list
val hstack : t list -> t
val hypot : t -> t -> t

val igamma : t -> t -> t
val igammac : t -> t -> t
val imag : t -> t
val index : t -> indices:t option list -> t
val index_add : t -> dim:int -> index:t -> source:t -> t
val index_copy : t -> dim:int -> index:t -> source:t -> t
val index_fill : t -> dim:int -> index:t -> value:'a Scalar.t -> t
val index_put : ?accumulate:bool -> indices:t option list -> values:t -> t -> t
val index_select : t -> dim:int -> index:t -> t
val indices : t -> t
val inner : t -> t -> t
val inverse : t -> t
val isclose : ?rtol:float -> ?atol:float -> ?equal_nan:bool -> t -> t -> t
val isfinite : t -> t
val isinf : t -> t
val isnan : t -> t
val isneginf : t -> t
val isposinf : t -> t
val isreal : t -> t

val kl_div : ?reduction:Torch_core.Reduction.t -> ?log_target:bool -> target:t -> t -> t
val kron : t -> t -> t

val l1_loss : ?reduction:Torch_core.Reduction.t -> target:t -> t -> t
val lcm : t -> t -> t
val ldexp : t -> t -> t
val le : t -> 'a Scalar.t -> t
val le_tensor : t -> t -> t
val leaky_relu : t -> t
val lerp : t -> end_:t -> weight:'a Scalar.t -> t
val lerp_tensor : t -> end_:t -> weight:t -> t
val lgamma : t -> t
val linalg_cholesky : ?upper:bool -> t -> t
val linalg_cholesky_ex : ?upper:bool -> ?check_errors:bool -> t -> t * t
val linalg_det : t -> t
val linalg_eig : t -> t * t
val linalg_eigh : ?uplo:string -> t -> t * t
val linalg_eigvals : t -> t
val linalg_eigvalsh : ?uplo:string -> t -> t
val linalg_inv : t -> t
val linalg_inv_ex : ?check_errors:bool -> t -> t * t
val linalg_matrix_power : t -> n:int -> t
val linalg_multi_dot : t list -> t
val linalg_pinv : ?rcond:float -> ?hermitian:bool -> t -> t
val linalg_qr : ?mode:string -> t -> t * t
val linalg_slogdet : t -> t * t
val linalg_solve : t -> t -> t
val linalg_svd : ?full_matrices:bool -> t -> t * t * t
val linalg_svdvals : t -> t
val linalg_tensorinv : ?ind:int -> t -> t
val linalg_tensorsolve : ?dims:int list -> t -> t -> t
val linear : ?bias:t -> weight:t -> t -> t
val linspace : ?kind:kind -> ?device:Device.t -> start:'a Scalar.t -> end_:'a Scalar.t -> int -> t
val log : t -> t
val log10 : t -> t
val log1p : t -> t
val log2 : t -> t
val log_normal : ?device:Device.t -> mean:float -> std:float -> int list -> t
val log_sigmoid : t -> t
val logaddexp : t -> t -> t
val logaddexp2 : t -> t -> t
val logcumsumexp : t -> dim:int -> t
val logdet : t -> t
val logical_and : t -> t -> t
val logical_not : t -> t
val logical_or : t -> t -> t
val logical_xor : t -> t -> t
val logit : t -> eps:float -> t
val logsumexp : dim:int list -> ?keepdim:bool -> t -> t
val lt : t -> 'a Scalar.t -> t
val lt_tensor : t -> t -> t
val lu_solve : t -> lu_data:t -> lu_pivots:t -> t

val matmul : t -> t -> t
val matrix_exp : t -> t
val matrix_power : t -> n:int -> t
val matrix_rank : ?symmetric:bool -> t -> t
val max_dim : dim:int -> ?keepdim:bool -> t -> t * t
val median : t -> t
val mish : t -> t
val mm : t -> mat2:t -> t
val mse_loss : ?reduction:Torch_core.Reduction.t -> target:t -> t -> t
val msort : t -> t
val mul : t -> t -> t
val mul_scalar : t -> 'a Scalar.t -> t

(** [multinomial ~num_samples ?(replacement=false) t] returns a tensor where each row contains [num_samples] indices
    sampled from the multinomial probability distribution located in the corresponding row of tensor [t]. *)
val multinomial : num_samples:int -> ?replacement:bool -> t -> t

val mv : t -> vec:t -> t
val mvlgamma : t -> p:int -> t

val narrow : t -> dim:int -> start:int -> length:int -> t
val ne : t -> 'a Scalar.t -> t
val ne_tensor : t -> t -> t
val neg : t -> t
val nll_loss : ?weight:t -> ?reduction:Torch_core.Reduction.t -> ?ignore_index:int -> target:t -> t -> t
val norm : t -> t
val norm_scalaropt_dim : p:'a Scalar.t -> ?dim:int list -> ?keepdim:bool -> t -> t
val normal_tensor_tensor : mean:t -> std:t -> t

val one_hot : t -> num_classes:int -> t
val ones : ?kind:kind -> ?device:Device.t -> int list -> t
val ones_like : t -> t
val outer : t -> vec2:t -> t

val pairwise_distance : ?p:float -> ?eps:float -> ?keepdim:bool -> x1:t -> x2:t -> t
val pdist : ?p:float -> t -> t
val permute : t -> dims:int list -> t
val poisson : t -> t
val polar : abs:t -> angle:t -> t
val polygamma : n:int -> t -> t
val positive : t -> t
val pow : t -> exponent:t -> t
val pow_scalar : 'a Scalar.t -> exponent:t -> t
val pow_tensor_scalar : t -> exponent:'a Scalar.t -> t
val prelu : t -> weight:t -> t
val prod : ?kind:kind -> t -> t
val prod_dim_int : dim:int -> ?keepdim:bool -> ?kind:kind -> t -> t

val qr : ?some:bool -> t -> t * t

val rad2deg : t -> t

(** [rand shape] creates a tensor with random values sampled uniformly between 0 and 1. *)
val rand : ?kind:kind -> ?device:Device.t -> int list -> t

val rand_like : t -> t
val randint : ?kind:kind -> ?device:Device.t -> high:int -> int list -> t
val randint_like : t -> high:int -> t
val randint_like_low_dtype : t -> low:int -> high:int -> t
val randint_low : ?kind:kind -> ?device:Device.t -> low:int -> high:int -> int list -> t

(** [randn shape] creates a tensor with random values sampled using a standard normal distribution. *)
val randn : ?kind:kind -> ?device:Device.t -> int list -> t

val randn_like : t -> t
val randperm : ?kind:kind -> ?device:Device.t -> n:int -> t
val range : ?kind:kind -> ?device:Device.t -> start:'a Scalar.t -> 'a Scalar.t -> t
val range_step : ?kind:kind -> ?device:Device.t -> start:'a Scalar.t -> 'a Scalar.t -> t
val ravel : t -> t
val real : t -> t
val reciprocal : t -> t
val relu : t -> t
val relu6 : t -> t
val remainder : t -> 'a Scalar.t -> t
val remainder_scalar_tensor : 'a Scalar.t -> t -> t
val remainder_tensor : t -> t -> t
val renorm : t -> p:'a Scalar.t -> dim:int -> maxnorm:'a Scalar.t -> t
val repeat : t -> repeats:int list -> t
val reshape : t -> shape:int list -> t
val reshape_as : t -> t -> t
val resolve_conj : t -> t
val resolve_neg : t -> t
val roll : shifts:int list -> ?dims:int list -> t -> t
val rot90 : t -> k:int -> dims:int list -> t
val round : t -> t
val row_stack : t list -> t
val rrelu : ?training:bool -> t -> t
val rsqrt : t -> t
val rsub : t -> t -> t
val rsub_scalar : t -> 'a Scalar.t -> t

val scalar_tensor : ?kind:kind -> ?device:Device.t -> 'a Scalar.t -> t
val scatter : t -> dim:int -> index:t -> src:t -> t
val scatter_add : t -> dim:int -> index:t -> src:t -> t
val scatter_reduce : t -> dim:int -> index:t -> src:t -> reduce:string -> t
val scatter_value : t -> dim:int -> index:t -> value:'a Scalar.t -> t

(** [select t ~dim ~index] slices tensor [t] along the selected dimension [dim] at the given index [index]. *)
val select : t -> dim:int -> index:int -> t

val selu : t -> t
val set_requires_grad : t -> r:bool -> t
val sgn : t -> t
val sigmoid : t -> t
val signbit : t -> t
val silu : t -> t
val sin : t -> t
val sinc : t -> t
val sinh : t -> t
val slice : t -> dim:int -> start:int -> end_:int -> step:int -> t
val slogdet : t -> t * t
val smm : t -> mat2:t -> t
val smooth_l1_loss : target:t -> ?reduction:Torch_core.Reduction.t -> ?beta:float -> t -> t
val soft_margin_loss : target:t -> ?reduction:Torch_core.Reduction.t -> t -> t
val softmax : dim:int -> ?kind:kind -> t -> t
val softplus : t -> t
val softshrink : t -> t
val solve : t -> a:t -> t * t
val sort : ?dim:int -> ?descending:bool -> t -> t * t
val sort_stable : ?dim:int -> ?descending:bool -> t -> t * t
val sparse_mask : t -> mask:t -> t
val special_digamma : t -> t
val special_entr : t -> t
val special_erf : t -> t
val special_erfc : t -> t
val special_erfcx : t -> t
val special_erfinv : t -> t
val special_exp2 : t -> t
val special_expit : t -> t
val special_expm1 : t -> t
val special_gammainc : t -> t -> t
val special_gammaincc : t -> t -> t
val special_gammaln : t -> t
val special_i0 : t -> t
val special_i0e : t -> t
val special_i1 : t -> t
val special_i1e : t -> t
val special_log1p : t -> t
val special_log_softmax : dim:int -> ?kind:kind -> t -> t
val special_logit : t -> eps:float -> t
val special_logsumexp : dim:int list -> ?keepdim:bool -> t -> t
val special_multigammaln : t -> p:int -> t
val special_ndtr : t -> t
val special_ndtri : t -> t
val special_polygamma : n:int -> t -> t
val special_psi : t -> t
val special_round : t -> t
val special_sinc : t -> t
val special_xlog1py : t -> t -> t
val special_xlog1py_other_scalar : t -> 'a Scalar.t -> t
val special_xlog1py_self_scalar : 'a Scalar.t -> t -> t
val special_xlogy : t -> t -> t
val special_xlogy_other_scalar : t -> 'a Scalar.t -> t
val special_xlogy_self_scalar : 'a Scalar.t -> t -> t
val special_zeta : t -> t -> t
val special_zeta_other_scalar : t -> 'a Scalar.t -> t
val special_zeta_self_scalar : 'a Scalar.t -> t -> t
val split : ?dim:int -> split_size:int -> t -> t list
val split_with_sizes : ?dim:int -> split_sizes:int list -> t -> t list
val sqrt : t -> t
val square : t -> t
val squeeze : t -> t

(** [squeeze_dim t ~dim] returns a tensor with the dimension [dim] of [t] of size [1] removed. *)
val squeeze_dim : t -> dim:int -> t

(** [stack ?(dim=0) ts] concatenates a sequence of tensors along a new dimension. *)
val stack : ?dim:int -> t list -> t

val std : t -> unbiased:bool -> t
val sub : t -> t -> t
val sub_scalar : t -> 'a Scalar.t -> t

(** [sum_dim_intlist ~dim ?(keepdim=false) t] returns the sum of each row of the tensor [t] in the given
    dimension [dim]. If [dim] is a list of dimensions, reduce over all of them. *)
val sum_dim_intlist : dim:int list -> ?keepdim:bool -> ?kind:kind -> t -> t

val sum_to_size : t -> size:int list -> t
val svd : ?some:bool -> ?compute_uv:bool -> t -> t * t * t

val tr : t -> t
val take : t -> index:t -> t
val take_along_dim : t -> indices:t -> dim:int -> t
val tan : t -> t
val tanh : t -> t
val tensor_split : sections:int -> ?dim:int -> t -> t list
val tensor_split_indices : indices:int list -> ?dim:int -> t -> t list
val tensordot : t -> t -> dims_self:int list -> dims_other:int list -> t
val threshold : t -> threshold:'a Scalar.t -> value:'a Scalar.t -> t
val tile : t -> dims:int list -> t
val trace : t -> t
val transpose : t -> dim0:int -> dim1:int -> t
val trapezoid : ?dim:int -> y:t -> t
val trapezoid_x : ?dim:int -> y:t -> x:t -> t

(** [triangular_solve ~a ?(upper=true) ?(transpose=false) ?(unitriangular=false) t] solves a system
    of equations with a triangular coefficient matrix [a] and multiple right-hand sides [b]. *)
val triangular_solve : a:t -> ?upper:bool -> ?transpose:bool -> ?unitriangular:bool -> t -> t * t

(** [tril ?(diagonal=0) t] returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
    [t], the other elements of the result tensor are set to 0. *)
val tril : ?diagonal:int -> t -> t

val triu : ?diagonal:int -> t -> t
val true_divide : t -> t -> t
val true_divide_scalar : t -> 'a Scalar.t -> t

val trunc : t -> t
val type_as : t -> t -> t

val unbind : ?dim:int -> t -> t list
val unfold : t -> dimension:int -> size:int -> step:int -> t
val unflatten : t -> dim:int -> sizes:int list -> t
val uniform : ?kind:kind -> ?device:Device.t -> from:float -> to_:float -> int list -> t

(** [unsqueeze t ~dim] returns a new tensor with a dimension of size one inserted at the
    specified position. *)
val unsqueeze : t -> dim:int -> t

val values : t -> t
val var : t -> unbiased:bool -> t
val vdot : t -> t -> t
val view : t -> size:int list -> t
val view_as : t -> t -> t
val view_as_complex : t -> t
val view_as_real : t -> t
val view_kind : t -> kind:kind -> t
val vsplit : t -> sections:int -> t list
val vsplit_array : t -> indices:int list -> t list
val vstack : t list -> t

val where : condition:t -> t list
val where_scalar : condition:t -> 'a Scalar.t -> 'a Scalar.t -> t
val where_scalarother : condition:t -> t -> 'a Scalar.t -> t
val where_scalarself : condition:t -> 'a Scalar.t -> t -> t
val where_self : condition:t -> t -> t -> t

val xlogy : t -> t -> t
val xlogy_scalar_other : t -> 'a Scalar.t -> t
val xlogy_scalar_self : 'a Scalar.t -> t -> t

val zeros : ?kind:kind -> ?device:Device.t -> int list -> t
val zeros_like : t -> t

(** Convenient APIs *)

val bool_vec : ?device:Device.t -> bool list -> t
val int_vec : ?device:Device.t -> int list -> t
val float_vec : ?device:Device.t -> float list -> t
val shape : t -> int list
val shape1_exn : t -> int
val shape2_exn : t -> int * int
val shape3_exn : t -> int * int * int
val shape4_exn : t -> int * int * int * int
val requires_grad : t -> bool
val grad_set_enabled : bool -> bool

val get : t -> int -> t
val bool_value : t -> bool
val int_value : t -> int
val float_value : t -> float
val bool_get : t -> int list -> bool
val int_get : t -> int list -> int
val float_get : t -> int list -> float
val bool_set : t -> int list -> bool -> unit
val int_set : t -> int list -> int -> unit
val float_set : t -> int list -> float -> unit
val fill_bool : t -> bool -> unit
val fill_int : t -> int -> unit
val fill_float : t -> float -> unit
val backward : ?keep_graph:bool -> ?create_graph:bool -> t -> unit
val run_backward : ?keep_graph:bool -> ?create_graph:bool -> t list -> t list -> t list
val print : t -> unit
val sum : t -> t
val mean : t -> t
val device : t -> Device.t
val max : t -> t -> t
val min : t -> t -> t

val ( .%{} ) : t -> int list -> int
val ( .%.{} ) : t -> int list -> float
val ( .%[] ) : t -> int -> int
val ( .%.[] ) : t -> int -> float
val ( .|[] ) : t -> int -> t

val no_grad : (unit -> 'a) -> 'a
val no_grad_ : t -> f:(t -> 'a) -> 'a
val zero_grad : t -> unit

val ( + ) : t -> t -> t
val ( - ) : t -> t -> t
val ( * ) : t -> t -> t
val ( / ) : t -> t -> t
val ( += ) : t -> t -> unit
val ( -= ) : t -> t -> unit
val ( *= ) : t -> t -> unit
val ( /= ) : t -> t -> unit
val ( ~- ) : t -> t

val ( = ) : t -> t -> t
val ( <> ) : t -> t -> t
val ( < ) : t -> t -> t
val ( <= ) : t -> t -> t
val ( > ) : t -> t -> t
val ( >= ) : t -> t -> t

val b : bool -> t
val i : int -> t
val f : float -> t

type ('a, 'b) create =
  ?requires_grad:bool
  -> ?kind:kind
  -> ?device:Device.t
  -> ?scale:'a Scalar.t
  -> 'b
  -> t

val new_zeros : ('a, int list) create
val new_ones : ('a, int list) create
val new_rand : ('a, int list) create
val new_randn : ('a, int list) create
val new_eye : ('a, int) create

val output : Stdio.Out_channel.t -> t -> unit

val copy : t -> t

val to_float0_exn : t -> float
val to_float1_exn : t -> float array
val to_float2_exn : t -> float array array
val to_float3_exn : t -> float array array array
val to_int0_exn : t -> int
val to_int1_exn : t -> int array
val to_int2_exn : t -> int array array
val to_int3_exn : t -> int array array array

val of_float0 : ?device:Device.t -> float -> t
val of_float1 : ?device:Device.t -> float array -> t
val of_float2 : ?device:Device.t -> float array array -> t
val of_float3 : ?device:Device.t -> float array array array -> t
val of_int0 : ?device:Device.t -> int -> t
val of_int1 : ?device:Device.t -> int array -> t
val of_int2 : ?device:Device.t -> int array array -> t
val of_int3 : ?device:Device.t -> int array array array -> t

val pp : Formatter.t -> t -> unit
[@@ocaml.toplevel_printer]

val maximum : t -> t
val minimum : t -> t

val scale_i : t -> int -> t
val scale_f : t -> float -> t

val to_list : t -> t list option
val to_list_exn : t -> t list

val reverse : t -> t
