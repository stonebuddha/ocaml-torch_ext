open Base
open Torch
module W = Torch_core.Wrapper.Tensor

type t = Tensor.t

let equal = Tensor.eq

(** Constants *)

let float_eps = 1e-07
let float_tiny = 1e-37

(** Simplified Kinds *)

type kind =
  | Bool
  | Int
  | Float
  | Complex
[@@deriving equal]

let string_of_kind = function
  | Bool -> "bool"
  | Int -> "int"
  | Float -> "float"
  | Complex -> "complex"

let to_packed = function
  | Bool -> Torch_core.Kind.T Bool
  | Int -> T Int
  | Float -> T Float
  | Complex -> T ComplexFloat

let to_kind t ~kind = W.totype t ~scalar_type:(to_packed kind)

let kind t =
  match W.kind t with
  | T Bool -> Bool
  | T Uint8
  | T Int8
  | T Int16
  | T Int
  | T Int64 -> Int
  | T Half
  | T Float
  | T Double -> Float
  | T ComplexHalf
  | T ComplexFloat
  | T ComplexDouble -> Complex

(** Generated APIs *)

let _sample_dirichlet = W._sample_dirichlet

let _standard_gamma = W._standard_gamma

let abs = W.abs

let acos = W.acos

let acosh = W.acosh

let add = W.add

let add_scalar = W.add_scalar

let all = W.all

let all_dim ~dim ?(keepdim=false) t = W.all_dim t ~dim ~keepdim

let alpha_dropout = W.alpha_dropout

let amax ~dim ?(keepdim=false) t = W.amax t ~dim ~keepdim

let amin ~dim ?(keepdim=false) t = W.amin t ~dim ~keepdim

let angle = W.angle

let any = W.any

let any_dim ~dim ?(keepdim=false) t = W.any_dim t ~dim ~keepdim

let arange ?(kind=Float) ?(device=Device.Cpu) end_ =
  W.arange ~end_ ~options:(to_packed kind, device)

let arange_start ?(kind=Float) ?(device=Device.Cpu) ~start end_ =
  W.arange_start ~start ~end_ ~options:(to_packed kind, device)

let arange_start_step ?(kind=Float) ?(device=Device.Cpu) ~start ~step end_ =
  W.arange_start_step ~start ~end_ ~step ~options:(to_packed kind, device)

let argmax ~dim ?(keepdim=false) t = W.argmax ~dim ~keepdim t

let argmin ~dim ?(keepdim=false) t = W.argmin t ~dim ~keepdim

let argsort ?(dim=(-1)) ?(descending=false) t = W.argsort t ~dim ~descending

let as_strided ~size ~stride ?(storage_offset=0) t = W.as_strided t ~size ~stride ~storage_offset

let asin = W.asin

let asinh = W.asinh

let atan = W.atan

let atan2 = W.atan2

let atanh = W.atanh

let bernoulli = W.bernoulli

let bernoulli_float ?(device=Device.Cpu) ~p size =
  let t = W.empty ~size ~options:(T Float, device) in
  W.bernoulli_float_ t ~p

let bilinear ~weight ?bias ~input1 ~input2 =
  W.bilinear ~input1 ~input2 ~weight ~bias

let binary_cross_entropy ~target ?weight ?(reduction=Torch_core.Reduction.Elementwise_mean) t =
  W.binary_cross_entropy t ~target ~weight ~reduction

let binary_cross_entropy_with_logits ~target ?weight ?pos_weight ?(reduction=Torch_core.Reduction.Elementwise_mean) t =
  W.binary_cross_entropy_with_logits t ~target ~weight ~pos_weight ~reduction

let bincount ?weights ?(minlength=0) t = W.bincount t ~weights ~minlength

let binomial = W.binomial

let bitwise_and = W.bitwise_and

let bitwise_and_tensor = W.bitwise_and_tensor

let bitwise_not = W.bitwise_not

let bitwise_or = W.bitwise_or

let bitwise_or_tensor = W.bitwise_or_tensor

let bitwise_xor = W.bitwise_xor

let bitwise_xor_tensor = W.bitwise_xor_tensor

let block_diag = W.block_diag

let broadcast_tensors = W.broadcast_tensors

let broadcast_to = W.broadcast_to

let cartesian_prod = W.cartesian_prod

let cat ?(dim=0) ts = W.cat ts ~dim

let cauchy ?(device=Device.Cpu) ~median ~sigma size =
  let t = W.empty ~size ~options:(T Float, device) in
  W.cauchy_ t ~median ~sigma

let ceil = W.ceil

let celu = W.celu

let chain_matmul = W.chain_matmul

let cholesky ?(upper=false) t = W.cholesky t ~upper

let cholesky_inverse ?(upper=false) t = W.cholesky_inverse t ~upper

let cholesky_solve ?(upper=false) ~input2 t = W.cholesky_solve t ~input2 ~upper

let chunk ~chunks ?(dim=0) t = W.chunk t ~chunks ~dim

let clamp = W.clamp

let clamp_max = W.clamp_max

let clamp_max_tensor = W.clamp_max_tensor

let clamp_min = W.clamp_min

let clamp_min_tensor = W.clamp_min_tensor

let clamp_tensor ?min ?max t = W.clamp_tensor t ~min ~max

let clone = W.clone

let column_stack = W.column_stack

let combinations ?(r=2) ?(with_replacement=false) t = W.combinations t ~r ~with_replacement

let complex = W.complex

let conj = W.conj

let corrcoef = W.corrcoef

let cos = W.cos

let cosh = W.cosh

let cosine_similarity ?(dim=1) ?(eps=1e-08) x1 x2 = W.cosine_similarity ~x1 ~x2 ~dim ~eps

let cross_entropy_loss ~target ?weight ?(reduction=Torch_core.Reduction.Elementwise_mean) ?(ignore_index=(-100)) ?(label_smoothing=0.0) t =
  W.cross_entropy_loss t ~target ~weight ~reduction ~ignore_index ~label_smoothing

let cummax = W.cummax

let cummin = W.cummin

let cumprod ~dim ?kind t =
  let dtype =
    match kind with
    | Some kind -> to_packed kind
    | None -> W.kind t
  in
  W.cumprod t ~dim ~dtype

let cumsum ~dim ?kind t =
  let dtype =
    match kind with
    | Some kind -> to_packed kind
    | None -> W.kind t
  in
  W.cumsum t ~dim ~dtype

let data = W.data

let deg2rad = W.deg2rad

let det = W.det

let detach = W.detach

let diag ?(diagonal=0) t = W.diag t ~diagonal

let diag_embed ?(offset=0) ?(dim1=(-2)) ?(dim2=(-1)) t =
  W.diag_embed t ~offset ~dim1 ~dim2

let diagflat ?(offset=0) t = W.diagflat t ~offset

let diagonal ?(offset=0) ?(dim1=0) ?(dim2=1) t =
  W.diagonal t ~offset ~dim1 ~dim2

let diff ?(n=1) ?(dim=(-1)) ?prepend ?append t = W.diff t ~n ~dim ~prepend ~append

let digamma = W.digamma

let dist = W.dist

let div = W.div

let div_scalar = W.div_scalar

let dot = W.dot

let dropout ?(p=0.5) ?(train=true) t = W.dropout t ~p ~train

let dsplit = W.dsplit

let dsplit_array = W.dsplit_array

let dstack = W.dstack

let eig ?(eigenvectors=false) t = W.eig t ~eigenvectors

let einsum = W.einsum

let elu = W.elu

let empty ?(kind=Float) ?(device=Device.Cpu) ~size =
  W.empty ~size ~options:(to_packed kind, device)

let empty_like = W.empty_like

let eq = W.eq

let eq_tensor = W.eq_tensor

let erf = W.erf

let erfc = W.erfc

let erfinv = W.erfinv

let exp = W.exp

let exp2 = W.exp2

let expand ~size ?(implicit=false) t =
  W.expand t ~size ~implicit

let expand_as = W.expand_as

let expm1 = W.expm1

let eye ?(kind=Float) ?(device=Device.Cpu) n = W.eye ~n ~options:(to_packed kind, device)

let eye_m ?(kind=Float) ?(device=Device.Cpu) ~n ~m = W.eye_m ~n ~m ~options:(to_packed kind, device)

let fill_ = W.fill_

let fill_diagonal_ ?(wrap=false) ~fill_value t = W.fill_diagonal_ t ~fill_value ~wrap

let fill_tensor_ = W.fill_tensor_

let flatten ?(start_dim=0) ?(end_dim=(-1)) t = W.flatten t ~start_dim ~end_dim

let flip = W.flip

let fliplr = W.fliplr

let flipud = W.flipud

let float_power = W.float_power

let float_power_scalar = W.float_power_scalar

let float_power_tensor_scalar = W.float_power_tensor_scalar

let floor = W.floor

let floor_divide = W.floor_divide

let floor_divide_scalar = W.floor_divide_scalar

let fmax = W.fmax

let fmin = W.fmin

let fmod = W.fmod

let fmod_tensor = W.fmod_tensor

let frac = W.frac

let frexp = W.frexp

let frobenius_norm = W.frobenius_norm

let full ?(kind=Float) ?(device=Device.Cpu) ~fill_value size =
  W.full ~size ~fill_value ~options:(to_packed kind, device)

let full_like = W.full_like

let gather ~dim ~index ?(sparse_grad=false) t =
  let index = W.totype index ~scalar_type:(T Int64) in
  W.gather t ~dim ~index ~sparse_grad

let gcd = W.gcd

let ge = W.ge

let ge_tensor = W.ge_tensor

let gelu = W.gelu

let geometric ?(device=Device.Cpu) ~p size =
  let t = W.empty ~size ~options:(T Float, device) in
  W.geometric_ t ~p

let grad = W.grad

let gt = W.gt

let gt_tensor = W.gt_tensor

let hinge_embedding_loss ~target ?(margin=1.0) ?(reduction=Torch_core.Reduction.Elementwise_mean) t =
  W.hinge_embedding_loss t ~target ~margin ~reduction

let histc = W.histc

let hsplit = W.hsplit

let hsplit_array = W.hsplit_array

let hstack = W.hstack

let hypot = W.hypot

let igamma = W.igamma

let igammac = W.igammac

let imag = W.imag

let index = W.index

let index_add = W.index_add

let index_copy = W.index_copy

let index_fill = W.index_fill

let index_put ?(accumulate=false) ~indices ~values t = W.index_put t ~indices ~values ~accumulate

let index_select t ~dim ~index =
  let index = W.totype index ~scalar_type:(T Int64) in
  W.index_select t ~dim ~index

let indices = W.indices

let inner = W.inner

let inverse = W.inverse

let isclose ?(rtol=1e-05) ?(atol=1e-08) ?(equal_nan=false) t1 t2 =
  W.isclose t1 t2 ~rtol ~atol ~equal_nan

let isfinite = W.isfinite

let isinf = W.isinf

let isnan = W.isnan

let isneginf = W.isneginf

let isposinf = W.isposinf

let isreal = W.isreal

let kl_div ?(reduction=Torch_core.Reduction.Elementwise_mean) ?(log_target=false) ~target t =
  W.kl_div t ~target ~reduction ~log_target

let kron = W.kron

let l1_loss ?(reduction=Torch_core.Reduction.Elementwise_mean) ~target t =
  W.l1_loss t ~reduction ~target

let lcm = W.lcm

let ldexp = W.ldexp

let le = W.le

let le_tensor = W.le_tensor

let leaky_relu = W.leaky_relu

let lerp = W.lerp

let lerp_tensor = W.lerp_tensor

let lgamma = W.lgamma

let linalg_cholesky ?(upper=false) t = W.linalg_cholesky t ~upper

let linalg_cholesky_ex ?(upper=false) ?(check_errors=false) t = W.linalg_cholesky_ex t ~upper ~check_errors

let linalg_det = W.linalg_det

let linalg_eig = W.linalg_eig

let linalg_eigh ?(uplo="L") t = W.linalg_eigh t ~uplo

let linalg_eigvals = W.linalg_eigvals

let linalg_eigvalsh ?(uplo="L") t = W.linalg_eigvalsh t ~uplo

let linalg_inv = W.linalg_inv

let linalg_inv_ex ?(check_errors=false) t = W.linalg_inv_ex t ~check_errors

let linalg_matrix_power = W.linalg_matrix_power

let linalg_multi_dot = W.linalg_multi_dot

let linalg_pinv ?(rcond=1e-15) ?(hermitian=false) t = W.linalg_pinv t ~rcond ~hermitian

let linalg_qr ?(mode="reduced") t = W.linalg_qr t ~mode

let linalg_slogdet = W.linalg_slogdet

let linalg_solve = W.linalg_solve

let linalg_svd ?(full_matrices=true) t = W.linalg_svd t ~full_matrices

let linalg_svdvals = W.linalg_svdvals

let linalg_tensorinv ?(ind=2) t = W.linalg_tensorinv t ~ind

let linalg_tensorsolve ?(dims=[]) t1 t2 = W.linalg_tensorsolve t1 t2 ~dims

let linear ?bias ~weight t = W.linear t ~weight ~bias

let linspace ?(kind=Float) ?(device=Device.Cpu) ~start ~end_ steps =
  W.linspace ~start ~end_ ~steps ~options:(to_packed kind, device)

let log = W.log

let log10 = W.log10

let log1p = W.log1p

let log2 = W.log2

let log_normal ?(device=Device.Cpu) ~mean ~std size =
  let t = W.empty ~size ~options:(T Float, device) in
  W.log_normal_ t ~mean ~std

let log_sigmoid = W.log_sigmoid

let logaddexp = W.logaddexp

let logaddexp2 = W.logaddexp2

let logcumsumexp = W.logcumsumexp

let logdet = W.logdet

let logical_and = W.logical_and

let logical_not = W.logical_not

let logical_or = W.logical_or

let logical_xor = W.logical_xor

let logit = W.logit

let logsumexp ~dim ?(keepdim=false) t = W.logsumexp t ~dim ~keepdim

let lt = W.lt

let lt_tensor = W.lt_tensor

let lu_solve = W.lu_solve

let matmul = W.matmul

let matrix_exp = W.matrix_exp

let matrix_power = W.matrix_power

let matrix_rank ?(symmetric=false) t = W.matrix_rank t ~symmetric

let max_dim ~dim ?(keepdim=false) t =
  let t1, t2 = W.max_dim t ~dim ~keepdim in
  t1, W.totype t2 ~scalar_type:(T Int)

let median = W.median

let mish = W.mish

let mm = W.mm

let mse_loss ?(reduction=Torch_core.Reduction.Elementwise_mean) ~target t =
  W.mse_loss t ~target ~reduction

let msort = W.msort

let mul = W.mul

let mul_scalar = W.mul_scalar

let multinomial ~num_samples ?(replacement=false) t =
  W.multinomial t ~num_samples ~replacement

let mv = W.mv

let mvlgamma = W.mvlgamma

let narrow = W.narrow

let ne = W.ne

let ne_tensor = W.ne_tensor

let neg = W.neg

let nll_loss ?weight ?(reduction=Torch_core.Reduction.Elementwise_mean) ?(ignore_index=(-100)) ~target t =
  W.nll_loss t ~target ~weight ~reduction ~ignore_index

let norm = W.norm

let norm_scalaropt_dim ~p ?(dim=[]) ?(keepdim=false) t =
  W.norm_scalaropt_dim t ~p ~dim ~keepdim

let normal_tensor_tensor ~mean ~std =
  let res = W.empty_like mean in
  W.normal_tensor_tensor_out ~out:res ~mean ~std

let one_hot = W.one_hot

let ones ?(kind=Float) ?(device=Device.Cpu) size =
  W.ones ~size ~options:(to_packed kind, device)

let ones_like = W.ones_like

let outer = W.outer

let pairwise_distance ?(p=2.0) ?(eps=1e-06) ?(keepdim=false) ~x1 ~x2 =
  W.pairwise_distance ~x1 ~x2 ~p ~eps ~keepdim

let pdist ?(p=2.0) t = W.pdist t ~p

let permute = W.permute

let poisson = W.poisson

let polar = W.polar

let polygamma = W.polygamma

let positive = W.positive

let pow = W.pow

let pow_scalar = W.pow_scalar

let pow_tensor_scalar = W.pow_tensor_scalar

let prelu = W.prelu

let prod ?(kind=Float) t = W.prod t ~dtype:(to_packed kind)

let prod_dim_int ~dim ?(keepdim=false) ?(kind=Float) t = W.prod_dim_int t ~dim ~keepdim ~dtype:(to_packed kind)

let qr ?(some=true) t = W.qr t ~some

let rad2deg = W.rad2deg

let rand ?(kind=Float) ?(device=Device.Cpu) size = W.rand ~size ~options:(to_packed kind, device)

let rand_like = W.rand_like

let randint ?(kind=Float) ?(device=Device.Cpu) ~high size = W.randint ~high ~size ~options:(to_packed kind, device)

let randint_like = W.randint_like

let randint_like_low_dtype = W.randint_like_low_dtype

let randint_low ?(kind=Float) ?(device=Device.Cpu) ~low ~high size =
  W.randint_low ~low ~high ~size ~options:(to_packed kind, device)

let randn ?(kind=Float) ?(device=Device.Cpu) size = W.randn ~size ~options:(to_packed kind, device)

let randn_like = W.randn_like

let randperm ?(kind=Int) ?(device=Device.Cpu) ~n = W.randperm ~n ~options:(to_packed kind, device)

let range ?(kind=Float) ?(device=Device.Cpu) ~start end_ =
  W.range ~start ~end_ ~options:(to_packed kind, device)

let range_step ?(kind=Float) ?(device=Device.Cpu) ~start end_ =
  W.range_step ~start ~end_ ~options:(to_packed kind, device)

let ravel = W.ravel

let real = W.real

let reciprocal = W.reciprocal

let relu = W.relu

let relu6 = W.relu6

let remainder = W.remainder

let remainder_scalar_tensor = W.remainder_scalar_tensor

let remainder_tensor = W.remainder_tensor

let renorm = W.renorm

let repeat = W.repeat

let reshape = W.reshape

let reshape_as = W.reshape_as

let resolve_conj = W.resolve_conj

let resolve_neg = W.resolve_neg

let roll ~shifts ?(dims=[]) t = W.roll t ~shifts ~dims

let rot90 = W.rot90

let round = W.round

let row_stack = W.row_stack

let rrelu ?(training=false) t = W.rrelu t ~training

let rsqrt = W.rsqrt

let rsub = W.rsub

let rsub_scalar = W.rsub_scalar

let scalar_tensor ?(kind=Float) ?(device=Device.Cpu) s = W.scalar_tensor ~s ~options:(to_packed kind, device)

let scatter = W.scatter

let scatter_add = W.scatter_add

let scatter_reduce = W.scatter_reduce

let scatter_value = W.scatter_value

let select = W.select

let selu = W.selu

let set_requires_grad = W.set_requires_grad

let sgn = W.sgn

let sigmoid = W.sigmoid

let signbit = W.signbit

let silu = W.silu

let sin = W.sin

let sinc = W.sinc

let sinh = W.sinh

let slice = W.slice

let slogdet = W.slogdet

let smm = W.smm

let smooth_l1_loss ~target ?(reduction=Torch_core.Reduction.Elementwise_mean) ?(beta=1.0) t =
  W.smooth_l1_loss t ~target ~reduction ~beta

let soft_margin_loss ~target ?(reduction=Torch_core.Reduction.Elementwise_mean) t =
  W.soft_margin_loss t ~target ~reduction

let softmax ~dim ?kind t =
  let dtype =
    match kind with
    | Some kind -> to_packed kind
    | None -> W.kind t
  in
  W.softmax t ~dim ~dtype

let softplus = W.softplus

let softshrink = W.softshrink

let solve = W.solve

let sort ?(dim=(-1)) ?(descending=false) t = W.sort t ~dim ~descending

let sort_stable ?(dim=(-1)) ?(descending=false) t = W.sort_stable t ~stable:true ~dim ~descending

let sparse_mask = W.sparse_mask

let special_digamma = W.special_digamma

let special_entr = W.special_entr

let special_erf = W.special_erf

let special_erfc = W.special_erfc

let special_erfcx = W.special_erfcx

let special_erfinv = W.special_erfinv

let special_exp2 = W.special_exp2

let special_expit = W.special_expit

let special_expm1 = W.special_expm1

let special_gammainc = W.special_gammainc

let special_gammaincc = W.special_gammaincc

let special_gammaln = W.special_gammaln

let special_i0 = W.special_i0

let special_i0e = W.special_i0e

let special_i1 = W.special_i1

let special_i1e = W.special_i1e

let special_log1p = W.special_log1p

let special_log_softmax ~dim ?kind t =
  let dtype =
    match kind with
    | Some kind -> to_packed kind
    | None -> W.kind t
  in
  W.special_log_softmax t ~dim ~dtype

let special_logit = W.special_logit

let special_logsumexp ~dim ?(keepdim=false) t = W.special_logsumexp t ~dim ~keepdim

let special_multigammaln = W.special_multigammaln

let special_ndtr = W.special_ndtr

let special_ndtri = W.special_ndtri

let special_polygamma = W.special_polygamma

let special_psi = W.special_psi

let special_round = W.special_round

let special_sinc = W.special_sinc

let special_xlog1py = W.special_xlog1py

let special_xlog1py_other_scalar = W.special_xlog1py_other_scalar

let special_xlog1py_self_scalar = W.special_xlog1py_self_scalar

let special_xlogy = W.special_xlogy

let special_xlogy_other_scalar = W.special_xlogy_other_scalar

let special_xlogy_self_scalar = W.special_xlogy_self_scalar

let special_zeta = W.special_zeta

let special_zeta_other_scalar = W.special_zeta_other_scalar

let special_zeta_self_scalar = W.special_zeta_self_scalar

let split ?(dim=0) ~split_size t = W.split t ~split_size ~dim

let split_with_sizes ?(dim=0) ~split_sizes t = W.split_with_sizes t ~split_sizes ~dim

let sqrt = W.sqrt

let square = W.square

let squeeze = W.squeeze

let squeeze_dim = W.squeeze_dim

let stack ?(dim=0) ts = W.stack ts ~dim

let std = W.std

let sub = W.sub

let sub_scalar = W.sub_scalar

let sum_dim_intlist ~dim ?(keepdim=false) ?kind t =
  let dtype =
    match kind with
    | Some kind -> to_packed kind
    | None -> W.kind t
  in
  W.sum_dim_intlist t ~dim ~keepdim ~dtype

let sum_to_size = W.sum_to_size

let svd ?(some=true) ?(compute_uv=true) t = W.svd t ~some ~compute_uv

let tr = W.tr

let take = W.take

let take_along_dim = W.take_along_dim

let tan = W.tan

let tanh = W.tanh

let tensor_split ~sections ?(dim=0) t = W.tensor_split t ~sections ~dim

let tensor_split_indices ~indices ?(dim=0) t = W.tensor_split_indices t ~indices ~dim

let tensordot = W.tensordot

let threshold = W.threshold

let tile = W.tile

let trace = W.trace

let transpose = W.transpose

let trapezoid ?(dim=(-1)) ~y = W.trapezoid ~y ~dim

let trapezoid_x ?(dim=(-1)) ~y ~x = W.trapezoid_x ~y ~x ~dim

let triangular_solve ~a ?(upper=true) ?(transpose=false) ?(unitriangular=false) t =
  W.triangular_solve t ~a ~upper ~transpose ~unitriangular

let tril ?(diagonal=0) t =
  W.tril t ~diagonal

let triu ?(diagonal=0) t = W.triu t ~diagonal

let true_divide = W.true_divide

let true_divide_scalar = W.true_divide_scalar

let trunc = W.trunc

let type_as = W.type_as

let unbind ?(dim=0) t = W.unbind t ~dim

let unfold = W.unfold

let unflatten = W.unflatten

let uniform ?(kind=Float) ?(device=Device.Cpu) ~from ~to_ size =
  let t = W.empty ~size ~options:(to_packed kind, device) in
  W.uniform_ t ~from ~to_

let unsqueeze = W.unsqueeze

let values = W.values

let var = W.var

let vdot = W.vdot

let view = W.view

let view_as = W.view_as

let view_as_complex = W.view_as_complex

let view_as_real = W.view_as_real

let view_kind t ~kind = W.view_dtype t ~dtype:(to_packed kind)

let vsplit = W.vsplit

let vsplit_array = W.vsplit_array

let vstack = W.vstack

let where = W.where

let where_scalar = W.where_scalar

let where_scalarother = W.where_scalarother

let where_scalarself = W.where_scalarself

let where_self = W.where_self

let xlogy = W.xlogy

let xlogy_scalar_other = W.xlogy_scalar_other

let xlogy_scalar_self = W.xlogy_scalar_self

let zeros ?(kind=Float) ?(device=Device.Cpu) size = W.zeros ~size ~options:(to_packed kind, device)

let zeros_like = W.zeros_like

(** Convenient APIs *)

let bool_vec ?device values =
  W.int_vec ~kind:`int (List.map values ~f:Bool.to_int)
  |> to_kind ~kind:Bool
  |> Tensor.to_device ?device

let int_vec ?device values =
  W.int_vec ~kind:`int values |> Tensor.to_device ?device

let float_vec ?device values =
  W.float_vec ~kind:`float values |> Tensor.to_device ?device

let shape = W.shape

let shape1_exn = W.shape1_exn

let shape2_exn = W.shape2_exn

let shape3_exn = W.shape3_exn

let shape4_exn = W.shape4_exn

let requires_grad = W.requires_grad

let grad_set_enabled = W.grad_set_enabled

let get = W.get

let bool_value t =
  W.int_value t
  |> Int.(( <> ) 0)

let int_value = W.int_value

let float_value = W.float_value

let bool_get t indexes =
  W.int_get t indexes
  |> Int.(( <> ) 0)

let int_get = W.int_get

let float_get = W.float_get

let bool_set t indexes b = W.int_set t indexes (Bool.to_int b)

let int_set = W.int_set

let float_set = W.float_set

let fill_bool t b = W.fill_int t (Bool.to_int b)

let fill_int = W.fill_int

let fill_float = W.fill_float

let backward = W.backward

let run_backward = W.run_backward

let print = W.print

let sum = W.sum

let mean = W.mean

let device = W.device

let max = W.max

let min = W.min

let ( .%{} ) = W.int_get
let ( .%.{} ) = W.float_get
let ( .%[] ) = Tensor.get_int1
let ( .%.[] ) = Tensor.get_float1
let ( .|[] ) = W.get

let no_grad = Tensor.no_grad

let no_grad_ = Tensor.no_grad_

let zero_grad = Tensor.zero_grad

let ( + ) = W.add

let ( - ) = W.sub

let ( * ) = W.mul

let ( / ) = W.div

let ( += ) = Tensor.( += )

let ( -= ) = Tensor.( -= )

let ( *= ) = Tensor.( *= )

let ( /= ) = Tensor.( /= )

let ( ~- ) = W.neg

let ( = ) = W.eq_tensor

let ( <> ) = W.ne_tensor

let ( < ) = W.lt_tensor

let ( <= ) = W.le_tensor

let ( > ) = W.gt_tensor

let ( >= ) = W.ge_tensor

let b v = bool_vec [v] |> W.reshape ~shape:[]

let i v = int_vec [v] |> W.reshape ~shape:[]

let f v = float_vec [v] |> W.reshape ~shape:[]

type ('a, 'b) create =
  ?requires_grad:bool
  -> ?kind:kind
  -> ?device:Device.t
  -> ?scale:'a Scalar.t
  -> 'b
  -> t

let gen
    ?(requires_grad=false)
    ?(kind=Float)
    ?(device=Device.Cpu)
    ?scale
    f
  =
  let t = f ~options:(to_packed kind, device) in
  let t =
    Option.value_map scale ~default:t ~f:(fun scale -> W.mul_scalar_ t scale)
  in
  if requires_grad then W.set_requires_grad t ~r:true else t

let new_zeros ?requires_grad ?kind ?device ?scale size =
  gen ?requires_grad ?kind ?device ?scale (W.zeros ~size)

let new_ones ?requires_grad ?kind ?device ?scale size =
  gen ?requires_grad ?kind ?device ?scale (W.ones ~size)

let new_rand ?requires_grad ?kind ?device ?scale size =
  gen ?requires_grad ?kind ?device ?scale (W.rand ~size)

let new_randn ?requires_grad ?kind ?device ?scale size =
  gen ?requires_grad ?kind ?device ?scale (W.randn ~size)

let new_eye ?requires_grad ?kind ?device ?scale n =
  gen ?requires_grad ?kind ?device ?scale (W.eye ~n)

let output ch t =
  let shape = shape t in
  let element_count = List.fold shape ~init:1 ~f:Int.( * ) in
  if Int.(element_count < 1_000) then
    begin
      Stdio.Out_channel.newline ch;
      Stdio.Out_channel.output_string ch (Tensor.to_string t ~line_size:96);
      Stdio.Out_channel.newline ch
    end
  else
    begin
      List.map shape ~f:Int.to_string
      |> String.concat ~sep:", "
      |> Printf.sprintf "Tensor<%s>"
      |> Stdio.Out_channel.output_string ch
    end

let copy = Tensor.copy

let to_float0_exn = Tensor.to_float0_exn
let to_float1_exn = Tensor.to_float1_exn
let to_float2_exn = Tensor.to_float2_exn
let to_float3_exn = Tensor.to_float3_exn
let to_int0_exn = Tensor.to_int0_exn
let to_int1_exn = Tensor.to_int1_exn
let to_int2_exn = Tensor.to_int2_exn
let to_int3_exn = Tensor.to_int3_exn

let of_float0 = Tensor.of_float0
let of_float1 = Tensor.of_float1
let of_float2 = Tensor.of_float2
let of_float3 = Tensor.of_float3
let of_int0 = Tensor.of_int0
let of_int1 = Tensor.of_int1
let of_int2 = Tensor.of_int2
let of_int3 = Tensor.of_int3

let pp = Tensor.pp

let maximum = Tensor.maximum
let minimum = Tensor.minimum

let scale_i t i = W.mul_scalar t (Scalar.i i)

let scale_f t f = W.mul_scalar t (Scalar.f f)

let to_list t =
  match shape t with
  | [] -> None
  | size :: _ -> Some (List.init size ~f:(get t))

let to_list_exn t = Option.value_exn (to_list t)

let reverse t = W.permute t ~dims:(List.range 0 (List.length (shape t)) |> List.rev)
