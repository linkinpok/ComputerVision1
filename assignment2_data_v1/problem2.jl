using Images
using LinearAlgebra
using PyPlot
using Printf
using Statistics

# Load images from the yale_faces directory and return a MxN data matrix,
# where M is the number of pixels per face image and N is the number of images.
# Also return the dimensions of a single face image and the number of all face images
function loadfaces()

  return data::Array{Float64,2},facedim::Array{Int},n::Int
end

# Apply principal component analysis on the data matrix.
# Return the eigenvectors of covariance matrix of the data, the corresponding eigenvalues,
# the one-dimensional mean data matrix and a cumulated variance vector in increasing order.
function computepca(data::Array{Float64,2})

  return U::Array{Float64,2},lambda::Array{Float64,1},mu::Array{Float64,2},cumvar::Array{Float64,1}
end

# Plot the cumulative variance of the principal components
function plotcumvar(cumvar::Array{Float64,1})

  return nothing::Nothing
end


# Compute required number of components to account for (at least) 75/99% of the variance
function computecomponents(cumvar::Array{Float64,1})

  return n75::Int,n99::Int
end


# Display the mean face and the first 10 Eigenfaces in a single figure
function showeigenfaces(U::Array{Float64,2},mu::Array{Float64,2},facedim::Array{Int})

  return nothing::Nothing
end


# Fetch a single face with given index out of the data matrix
function takeface(data::Array{Float64,2},facedim::Array{Int},n::Int)

  return face::Array{Float64,2}
end


# Project a given face into the low-dimensional space with a given number of principal
# components and reconstruct it afterwards
function computereconstruction(faceim::Array{Float64,2},U::Array{Float64,2},mu::Array{Float64,2},n::Int)

  return recon::Array{Float64,2}
end

# Display all reconstructed faces in a single figure
function showreconstructedfaces(faceim, f5, f15, f50, f150)

  return nothing::Nothing
end

# Problem 2: Eigenfaces

function problem2()
  # load data
  data,facedim,N = loadfaces()

  # compute PCA
  U,lambda,mu,cumvar = computepca(data)

  # plot cumulative variance
  plotcumvar(cumvar)

  # compute necessary components for 75% / 99% variance coverage
  n75,n99 = computecomponents(cumvar)
  println(@sprintf("Necssary components for 75%% variance coverage: %i", n75))
  println(@sprintf("Necssary components for 99%% variance coverage: %i", n99))

  # plot mean face and first 10 Eigenfaces
  showeigenfaces(U,mu,facedim)

  # get a random face
  faceim = takeface(data,facedim,rand(1:N))

  # reconstruct the face with 5, 15, 50, 150 principal components
  f5 = computereconstruction(faceim,U,mu,5)
  f15 = computereconstruction(faceim,U,mu,15)
  f50 = computereconstruction(faceim,U,mu,50)
  f150 = computereconstruction(faceim,U,mu,150)

  # display the reconstructed faces
  showreconstructedfaces(faceim, f5, f15, f50, f150)

  return
end
