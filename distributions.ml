open Base
open Or_error.Let_syntax
module Tensor = Tensor_ext
module Scalar = Torch.Scalar

class virtual ['a] t = object (self)
  method has_rsample : bool = false
  method rsample ?sample_shape () : 'a Or_error.t =
    (ignore (sample_shape : int list option); Or_error.unimplemented "rsample")
  method sample ?sample_shape () : 'a =
    Or_error.ok_exn (Tensor.no_grad (fun () -> self#rsample ?sample_shape ()))
  method virtual expand : batch_shape:int list -> 'a t
  method virtual log_prob : 'a -> Tensor.t
  method virtual batch_shape : int list
  method virtual event_shape : int list
  method private extended_shape sample_shape =
    List.concat [sample_shape; self#batch_shape; self#event_shape]
end

let clamp_probs probs =
  Tensor.clamp probs ~min:(Scalar.f Tensor.float_eps) ~max:(Scalar.f Float.(1. - Tensor.float_eps))

let probs_to_logits ~is_binary probs =
  let ps_clamped = clamp_probs probs in
  if is_binary then
    Tensor.(log ps_clamped - log1p (- ps_clamped))
  else
    Tensor.log ps_clamped

let logits_to_probs ~is_binary logits =
  if is_binary then
    Tensor.sigmoid logits
  else
    Tensor.softmax ~dim:(-1) logits

let rec bernoulli probs : Tensor.t t =
  let probs = Tensor.to_kind probs ~kind:Float in
  let lazy logits = lazy (probs_to_logits ~is_binary:true probs) in
  object (self)
    inherit [Tensor.t] t

    method sample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      Tensor.no_grad (fun () ->
          let t = Tensor.(bernoulli (expand ~size probs)) in
          Tensor.to_kind t ~kind:Bool
        )

    method expand ~batch_shape =
      let new_probs = Tensor.expand ~size:batch_shape probs in
      (bernoulli new_probs)

    method log_prob t =
      let t = Tensor.to_kind t ~kind:Float in
      Tensor.(- (Tensor.binary_cross_entropy_with_logits logits ~target:t ~reduction:None))

    method batch_shape = Tensor.shape probs

    method event_shape = []
  end

let rec normal loc scale : Tensor.t t =
  let loc = Tensor.to_kind loc ~kind:Float in
  let scale = Tensor.to_kind scale ~kind:Float in
  let [@warning "-8"] [loc; scale] = Tensor.broadcast_tensors [loc; scale] in
  let lazy variance_dbl = lazy (Tensor.scale_f (Tensor.square scale) 2.) in
  let lazy log_scale = lazy (Tensor.log scale) in
  let lazy shift = lazy (Tensor.f Float.(log (sqrt (2. * pi)))) in
  object (self)
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      let zeros = Tensor.zeros size in
      let ones = Tensor.ones size in
      let eps = Tensor.normal_tensor_tensor ~mean:zeros ~std:ones in
      Ok (Tensor.( + ) loc  (Tensor.( * ) eps scale))

    method sample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      Tensor.no_grad (fun () ->
          Tensor.(normal_tensor_tensor ~mean:(expand ~size loc) ~std:(expand ~size scale))
        )

    method expand ~batch_shape =
      let new_loc = Tensor.expand ~size:batch_shape loc in
      let new_scale = Tensor.expand ~size:batch_shape loc in
      (normal new_loc new_scale)

    method log_prob t =
      Tensor.((- (square (t - loc))) / variance_dbl - log_scale - shift)

    method batch_shape = Tensor.shape loc

    method event_shape = []
  end

let rec gamma concentration rate : Tensor.t t =
  let concentration = Tensor.to_kind concentration ~kind:Float in
  let rate = Tensor.to_kind rate ~kind:Float in
  let [@warning "-8"] [concentration; rate] = Tensor.broadcast_tensors [concentration; rate] in
  object (self)
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      let value = Tensor.(_standard_gamma (expand ~size concentration) / (expand ~size rate)) in
      let value_detached = Tensor.detach value in
      Ok (Tensor.clamp_min value_detached ~min:(Scalar.f Tensor.float_tiny))

    method expand ~batch_shape =
      let new_concentration = Tensor.expand ~size:batch_shape concentration in
      let new_rate = Tensor.expand ~size:batch_shape rate in
      (gamma new_concentration new_rate)

    method log_prob t =
      Tensor.(concentration * log rate + (concentration - f 1.) * log t - rate * t - lgamma concentration)

    method batch_shape = Tensor.shape concentration

    method event_shape = []
  end

let rec uniform low high : Tensor.t t =
  let low = Tensor.to_kind low ~kind:Float in
  let high = Tensor.to_kind high ~kind:Float in
  let [@warning "-8"] [low; high] = Tensor.broadcast_tensors [low; high] in
  object (self)
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      let eps = Tensor.rand size in
      Ok Tensor.(low + eps * (high - low))

    method expand ~batch_shape =
      let new_low = Tensor.expand ~size:batch_shape low in
      let new_high = Tensor.expand ~size:batch_shape high in
      (uniform new_low new_high)

    method log_prob t =
      let lb = Tensor.(type_as (low <= t) low) in
      let ub = Tensor.(type_as (high > t) low) in
      Tensor.(log (lb * ub) - log (high - low))

    method batch_shape = Tensor.shape low

    method event_shape = []
  end

let rec dirichlet concentration : Tensor.t t =
  let concentration = Tensor.to_kind concentration ~kind:Float in
  object (self)
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      Ok Tensor.(_sample_dirichlet (expand ~size concentration))

    method expand ~batch_shape =
      let new_concentration = Tensor.expand ~size:(batch_shape @ self#event_shape) concentration in
      (dirichlet new_concentration)

    method log_prob t =
      Tensor.(sum_dim_intlist ~dim:[(-1)] (log t * (concentration - f 1.)) +
              lgamma (sum_dim_intlist ~dim:[(-1)] concentration) -
              sum_dim_intlist ~dim:[(-1)] (lgamma concentration))

    method batch_shape = List.drop_last_exn (Tensor.shape concentration)

    method event_shape = [List.last_exn (Tensor.shape concentration)]
  end

let rec beta concentration1 concentration0 : Tensor.t t =
  let concentration1 = Tensor.to_kind concentration1 ~kind:Float in
  let concentration0 = Tensor.to_kind concentration0 ~kind:Float in
  let [@warning "-8"] [conc1; conc0] = Tensor.broadcast_tensors [concentration1; concentration0] in
  let lazy dirichlet = lazy (dirichlet (Tensor.stack [conc1; conc0] ~dim:(-1))) in
  object
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let%bind t = dirichlet#rsample ~sample_shape () in
      Ok (Tensor.select t ~dim:(-1) ~index:0)

    method expand ~batch_shape =
      let new_concentration1 = Tensor.expand ~size:batch_shape concentration1 in
      let new_concentration0 = Tensor.expand ~size:batch_shape concentration0 in
      (beta new_concentration1 new_concentration0)

    method log_prob t =
      let v = Tensor.(stack [t; f 1. - t] ~dim:(-1)) in
      dirichlet#log_prob v

    method batch_shape = Tensor.shape concentration1

    method event_shape = []
  end

let clamp_by_zero x =
  Tensor.((clamp_min x ~min:(Scalar.f 0.) + x - clamp_max x ~max:(Scalar.f 0.)) / f 2.)

let rec binomial total_count probs : Tensor.t t =
  let total = Tensor.(new_ones ~scale:(Scalar.i total_count) (shape probs)) in
  let probs = Tensor.to_kind probs ~kind:Float in
  let lazy logits = lazy (probs_to_logits ~is_binary:true probs) in
  object (self)
    inherit [Tensor.t] t

    method sample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      Tensor.no_grad (fun () ->
          Tensor.(binomial ~count:(expand ~size total) ~prob:(expand ~size probs))
          |> Tensor.to_kind ~kind:Int
        )

    method expand ~batch_shape =
      let new_probs = Tensor.expand ~size:batch_shape probs in
      (binomial total_count new_probs)

    method log_prob t =
      let log_factorial_n = Tensor.(lgamma (total + f 1.0)) in
      let log_factorial_k = Tensor.(lgamma (t + f 1.0)) in
      let log_factorial_nmk = Tensor.(lgamma (total - t + f 1.0)) in
      let normalize_term = Tensor.(
          total * (clamp_by_zero logits) +
          total * (log1p (exp (- (abs logits)))) -
          log_factorial_n
        )
      in
      Tensor.(t * logits - log_factorial_k - log_factorial_nmk - normalize_term)

    method batch_shape = Tensor.shape probs

    method event_shape = []
  end

let rec categorical probs : Tensor.t t =
  let probs = Tensor.to_kind probs ~kind:Float in
  let probs = Tensor.(probs / sum_dim_intlist probs ~dim:[(-1)] ~keepdim:true) in
  let lazy num_events = lazy (List.last_exn (Tensor.shape probs)) in
  let lazy logits = lazy (probs_to_logits ~is_binary:false probs) in
  object (self)
    inherit [Tensor.t] t

    method sample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      let probs_2d = Tensor.reshape probs ~shape:[(-1); num_events] in
      let samples_2d = Tensor.reverse (Tensor.multinomial probs_2d ~num_samples:(List.reduce_exn (1 :: sample_shape) ~f:( * )) ~replacement:true) in
      Tensor.reshape samples_2d ~shape:size
      |> Tensor.to_kind ~kind:Int

    method expand ~batch_shape =
      let new_probs = Tensor.expand ~size:(batch_shape @ [num_events]) probs in
      (categorical new_probs)

    method log_prob t =
      let t_un = Tensor.unsqueeze t ~dim:(-1) in
      let t_un = Tensor.select t_un ~dim:(-1) ~index:0 |> Tensor.unsqueeze ~dim:(-1) in
      Tensor.gather logits ~dim:(-1) ~index:t_un
      |> Tensor.squeeze_dim ~dim:(-1)

    method batch_shape = List.drop_last_exn (Tensor.shape probs)

    method event_shape = []
  end

let rec categorical_with_logits logits : Tensor.t t =
  let logits = Tensor.to_kind logits ~kind:Float in
  let logits = Tensor.(logits - logsumexp ~dim:[-1] ~keepdim:true logits) in
  let lazy num_events = lazy (List.last_exn (Tensor.shape logits)) in
  let lazy probs = lazy (logits_to_probs ~is_binary:false logits) in
  object (self)
    inherit [Tensor.t] t

    method sample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      let probs_2d = Tensor.reshape probs ~shape:[(-1); num_events] in
      let samples_2d = Tensor.reverse (Tensor.multinomial probs_2d ~num_samples:(List.reduce_exn (1 :: sample_shape) ~f:( * )) ~replacement:true) in
      Tensor.reshape samples_2d ~shape:size
      |> Tensor.to_kind ~kind:Int

    method expand ~batch_shape =
      let new_logits = Tensor.expand ~size:(batch_shape @ [num_events]) logits in
      (categorical_with_logits new_logits)

    method log_prob t =
      let t_un = Tensor.unsqueeze t ~dim:(-1) in
      let t_un = Tensor.select t_un ~dim:(-1) ~index:0 |> Tensor.unsqueeze ~dim:(-1) in
      Tensor.gather logits ~dim:(-1) ~index:t_un
      |> Tensor.squeeze_dim ~dim:(-1)

    method batch_shape = List.drop_last_exn (Tensor.shape probs)

    method event_shape = []
  end

let rec geometric probs : Tensor.t t =
  let probs = Tensor.to_kind probs ~kind:Float in
  object (self)
    inherit [Tensor.t] t

    method sample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      Tensor.no_grad (fun () ->
          let u = Tensor.(rand size) in
          let u = Tensor.clamp_min u ~min:(Scalar.f Tensor.float_tiny) in
          Tensor.(floor (log u / log1p (- probs)))
          |> Tensor.to_kind ~kind:Int
        )

    method expand ~batch_shape =
      let new_probs = Tensor.expand ~size:batch_shape probs in
      (geometric new_probs)

    method log_prob t =
      let condition = Tensor.(logical_and (probs = f 1.0) (t = f 0.0)) in
      let probs' = Tensor.where_self ~condition (Tensor.f 0.0) probs in
      Tensor.(t * log1p (- probs') + log probs)

    method batch_shape = Tensor.shape probs

    method event_shape = []
  end

let rec poisson rate : Tensor.t t =
  let rate = Tensor.to_kind rate ~kind:Float in
  object (self)
    inherit [Tensor.t] t

    method sample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      Tensor.no_grad (fun () ->
          Tensor.poisson (Tensor.expand ~size rate)
          |> Tensor.to_kind ~kind:Int
        )

    method expand ~batch_shape =
      let new_rate = Tensor.expand ~size:batch_shape rate in
      (poisson new_rate)

    method log_prob t =
      Tensor.(log rate * t - rate - lgamma (t + f 1.))

    method batch_shape = Tensor.shape rate

    method event_shape = []
  end

let batch_mahalanobis bL bx =
  let n = List.last_exn (Tensor.shape bx) in
  let bx_batch_shape = List.drop_last_exn (Tensor.shape bx) in

  let bx_batch_dims = List.length bx_batch_shape in
  let bL_batch_dims = List.length (Tensor.shape bL) - 2 in
  let outer_batch_dims = bx_batch_dims - bL_batch_dims in
  let old_batch_dims = outer_batch_dims + bL_batch_dims in
  let new_batch_dims = outer_batch_dims + 2 * bL_batch_dims in
  let bx_new_shape =
    List.fold (List.zip_exn (List.take (Tensor.shape bL) bL_batch_dims) (List.drop (Tensor.shape bx |> List.drop_last_exn) outer_batch_dims))
      ~init:(List.take (Tensor.shape bx) outer_batch_dims)
      ~f:(fun acc (sL, sx) ->
          List.append acc [sx / sL; sL]
        )
  in
  let bx_new_shape = List.append bx_new_shape [n] in
  let bx = Tensor.reshape ~shape:bx_new_shape bx in
  let permute_dims = List.concat [
      List.range 0 outer_batch_dims;
      List.range ~stride:2 outer_batch_dims new_batch_dims;
      List.range ~stride:2 (outer_batch_dims + 1) new_batch_dims;
      [new_batch_dims];
    ]
  in
  let bx = Tensor.permute ~dims:permute_dims bx in

  let flat_L = Tensor.reshape ~shape:[-1; n; n] bL in
  let flat_x = Tensor.reshape ~shape:[-1; List.hd_exn (Tensor.shape flat_L); n] bx in
  let flat_x_swap = Tensor.permute ~dims:[1; 2; 0] flat_x in
  let m_swap = Tensor.triangular_solve ~a:flat_L ~upper:false flat_x_swap |> fst |> Tensor.square |> Tensor.sum_dim_intlist ~dim:[(-2)] in
  let m = Tensor.tr m_swap in

  let permuted_m = Tensor.reshape ~shape:(List.drop_last_exn (Tensor.shape bx)) m in
  let permuted_inv_dims =
    List.fold (List.range 0 bL_batch_dims)
      ~init:(List.range 0 outer_batch_dims)
      ~f:(fun acc i ->
          List.append acc [outer_batch_dims + i; old_batch_dims + i]
        )
  in
  let reshaped_m = Tensor.permute ~dims:permuted_inv_dims permuted_m in
  Tensor.reshape ~shape:bx_batch_shape reshaped_m

let rec multivariate_normal loc scale_tril : Tensor.t t =
  let loc = Tensor.to_kind loc ~kind:Float in
  let scale_tril = Tensor.to_kind scale_tril ~kind:Float in
  let unbroadcasted_scale_tril = scale_tril in
  object (self)
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      let eps = Tensor.(normal_tensor_tensor ~mean:(zeros size) ~std:(ones size)) in
      Ok Tensor.(loc + squeeze_dim ~dim:(-1) (matmul unbroadcasted_scale_tril (unsqueeze eps ~dim:(-1))))

    method expand ~batch_shape =
      let new_loc = Tensor.expand ~size:(batch_shape @ self#event_shape) loc in
      let new_scale_tril = Tensor.expand ~size:(List.concat [batch_shape; self#event_shape; self#event_shape]) scale_tril in
      (multivariate_normal new_loc new_scale_tril)

    method log_prob t =
      let diff = Tensor.(t - loc) in
      let m = batch_mahalanobis unbroadcasted_scale_tril diff in
      let half_log_det =
        Tensor.(log (diagonal unbroadcasted_scale_tril ~dim1:(-2) ~dim2:(-1)) |> sum_dim_intlist ~dim:[(-1)])
      in
      Tensor.(f (-0.5) * (f Float.(of_int (List.hd_exn self#event_shape) * log (2. * pi)) + m) - half_log_det)

    method batch_shape = List.drop_last_exn (Tensor.shape loc)

    method event_shape = [List.last_exn (Tensor.shape loc)]
  end

let rec lkj_cholesky dim concentration : Tensor.t t =
  let concentration = Tensor.to_kind concentration ~kind:Float in
  let lazy beta = lazy (
    let marginal_conc = Tensor.(concentration + f Float.(0.5 * of_int Int.(dim - 2))) in
    let offset = Tensor.arange (Scalar.i (dim - 1)) in
    let offset = Tensor.cat [Tensor.float_vec [0.]; offset] in
    let beta_conc1 = Tensor.(offset + f 0.5) in
    let beta_conc0 = Tensor.(unsqueeze ~dim:(-1) marginal_conc - (f 0.5) * offset) in
    beta beta_conc1 beta_conc0
  )
  in
  object (self)
    inherit [Tensor.t] t

    method sample ?(sample_shape=[]) () =
      let y = beta#sample ~sample_shape () |> Tensor.unsqueeze ~dim:(-1) in
      let u_normal = Tensor.randn (self#extended_shape sample_shape) |> Tensor.tril ~diagonal:(-1) in
      let u_hypersphere =
        Tensor.(u_normal /
                norm_scalaropt_dim u_normal ~p:(frobenius_norm u_normal |> float_value |> Scalar.f) ~dim:[(-1)] ~keepdim:true)
      in
      let () =
        let first_row = Tensor.select u_hypersphere ~dim:(-2) ~index:0 in
        ignore (Tensor.fill_ first_row ~value:(Scalar.f 0.) : Tensor.t)
      in
      let w = Tensor.(sqrt y * u_hypersphere) in
      let diag_elems = Tensor.clamp_min ~min:(Scalar.f Tensor.float_tiny) Tensor.(f 1. - sum_dim_intlist (square w) ~dim:[(-1)])
                       |> Tensor.sqrt
      in
      Tensor.(w + diag_embed diag_elems)

    method expand ~batch_shape =
      let new_concentration = Tensor.expand ~size:batch_shape concentration in
      (lkj_cholesky dim new_concentration)

    method log_prob t =
      let diag_elems =
        Tensor.diagonal t ~dim1:(-1) ~dim2:(-2) |> Tensor.index_select ~dim:(-1) ~index:(Tensor.arange_start ~kind:Int ~start:(Scalar.i 1) Scalar.(i dim))
      in
      let order = Tensor.arange_start ~start:(Scalar.i 2) (Scalar.i (dim + 1)) in
      let order = Tensor.(unsqueeze ~dim:(-1) (f 2. * (concentration - f 1.)) + i dim - order) in
      let unnormalized_log_pdf = Tensor.sum_dim_intlist ~dim:[(-1)] Tensor.(order * log diag_elems) in
      let dm1 = dim - 1 in
      let alpha = Tensor.(concentration + f Float.(0.5 * of_int dm1)) in
      let denominator = Tensor.(lgamma alpha * i dm1) in
      let numerator = Tensor.(mvlgamma (alpha - f 0.5) ~p:dm1) in
      let pi_constant = Float.(0.5 * of_int dm1 * log pi) in
      let normalize_term = Tensor.(f pi_constant + numerator - denominator) in
      Tensor.(unnormalized_log_pdf - normalize_term)

    method batch_shape = Tensor.shape concentration

    method event_shape = [dim; dim]
  end

let sum_rightmost t dim =
  if dim = 0 then
    t
  else
    let shape = Tensor.shape t in
    let required_shape = List.take shape (List.length shape - dim) @ [-1] in
    Tensor.(reshape ~shape:required_shape t |> sum_dim_intlist ~dim:[-1])

let rec independent (base : Tensor.t t) reinterp_ndims : Tensor.t t =
  let shape = base#batch_shape @ base#event_shape in
  let event_dim = reinterp_ndims + List.length base#event_shape in
  let batch_shape, event_shape = List.split_n shape (List.length shape - event_dim) in
  object (self)
    inherit [Tensor.t] t

    method has_rsample = base#has_rsample

    method rsample ?sample_shape () =
      base#rsample ?sample_shape ()

    method sample ?sample_shape () =
      base#sample ?sample_shape ()

    method expand ~batch_shape =
      let new_base = base#expand ~batch_shape:(batch_shape @ List.take self#event_shape reinterp_ndims) in
      (independent new_base reinterp_ndims)

    method log_prob t =
      let log_prob = base#log_prob t in
      sum_rightmost log_prob reinterp_ndims

    method batch_shape = batch_shape

    method event_shape = event_shape
  end

let rec exponential rate : Tensor.t t =
  let rate = Tensor.to_kind rate ~kind:Float in
  object (self)
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      let u = Tensor.rand size in
      Ok Tensor.(- log1p (- u) / rate)

    method expand ~batch_shape =
      let new_rate = Tensor.expand ~size:batch_shape rate in
      (exponential new_rate)

    method log_prob t =
      Tensor.(log rate - rate * t)

    method batch_shape = Tensor.shape rate

    method event_shape = []
  end

let rec dirac v : Tensor.t t =
  object (self)
    inherit [Tensor.t] t

    method has_rsample = true

    method rsample ?(sample_shape=[]) () =
      let size = self#extended_shape sample_shape in
      Ok (Tensor.expand ~size v)

    method expand ~batch_shape =
      let new_v = Tensor.expand ~size:batch_shape v in
      (dirac new_v)

    method log_prob t =
      let lp = Tensor.(v = t |> to_kind ~kind:Float |> log) in
      sum_rightmost lp 0

    method batch_shape = Tensor.shape v

    method event_shape = []
  end
