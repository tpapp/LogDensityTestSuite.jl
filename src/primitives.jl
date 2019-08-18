#####
##### primitive distributions
#####

"Standard multivariate normal with the given dimension `K`."
struct StandardMultivariateNormal <: SamplingLogDensity
    K::Int
end

dimension(ℓ::StandardMultivariateNormal) = ℓ.K

logdensity(ℓ::StandardMultivariateNormal, x) = -0.5 * sum(abs2, x)

logdensity_and_gradient(ℓ::StandardMultivariateNormal, x) = logdensity(ℓ, x), -x

hypercube_transform(ℓ::StandardMultivariateNormal, x) = norminvcdf.(x)
