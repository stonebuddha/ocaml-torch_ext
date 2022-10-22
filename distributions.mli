open Base
open Torch

type 'a t = <
  has_rsample : bool;
  rsample : ?sample_shape:int list -> unit -> 'a Or_error.t;
  sample : ?sample_shape:int list -> unit -> 'a;
  expand : batch_shape:int list -> 'a t;
  log_prob : 'a -> Tensor.t;
  batch_shape : int list;
  event_shape : int list;
>

val bernoulli : Tensor.t -> Tensor.t t

val normal : Tensor.t -> Tensor.t -> Tensor.t t

val gamma : Tensor.t -> Tensor.t -> Tensor.t t

val uniform : Tensor.t -> Tensor.t -> Tensor.t t

val beta : Tensor.t -> Tensor.t -> Tensor.t t

val binomial : int -> Tensor.t -> Tensor.t t

val categorical : Tensor.t -> Tensor.t t

val categorical_with_logits : Tensor.t -> Tensor.t t

val geometric : Tensor.t -> Tensor.t t

val poisson : Tensor.t -> Tensor.t t

val dirichlet : Tensor.t -> Tensor.t t

val multivariate_normal : Tensor.t -> Tensor.t -> Tensor.t t

val lkj_cholesky : int -> Tensor.t -> Tensor.t t

val independent : Tensor.t t -> int -> Tensor.t t

val exponential : Tensor.t -> Tensor.t t

val dirac : Tensor.t -> Tensor.t t
