
# Negative Log-likelihood
type NLL <: mx.AbstractEvalMetric
  nll_sum  :: Float64
  n_sample :: Int

  NLL() = new(0.0, 0)
end

function mx.update!(metric :: NLL, labels :: Vector{mx.NDArray}, preds :: Vector{mx.NDArray})
  @assert length(labels) == length(preds)
  nll = 0.0
  for (label, pred) in zip(labels, preds)
    @mx.nd_as_jl ro=(label, pred) begin
      nll -= sum(log(max(broadcast_getindex(pred, round(Int,label+1), 1:length(label)), 1e-20)))
    end
  end

  nll = nll / length(labels)
  metric.nll_sum += nll
  metric.n_sample += length(labels[1])
end

function mx.get(metric :: NLL)
  nll  = metric.nll_sum / metric.n_sample
  perp = exp(nll)
  return [(:NLL, nll), (:perplexity, perp)]
end

function mx.reset!(metric :: NLL)
  metric.nll_sum  = 0.0
  metric.n_sample = 0
end
