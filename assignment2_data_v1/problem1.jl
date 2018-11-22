using Images
using PyPlot


# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)
  # define gaussian function
  gaussian(x,y,s)= 1/(2*pi*s*s) * exp(-(x*x+y*y)/(2*s*s))
  # define range in each dimension
  X = size[2]%2==1 ? (-floor(size[2]/2):floor(size[2]/2)) : (-size[2]/2+0.5:size[2]/2-0.5)
  Y = size[1]%2==1 ? (-floor(size[1]/2):floor(size[1]/2)) : (-size[1]/2+0.5:size[1]/2-0.5)
  # generate gaussian filter
  f = [gaussian(y,x,sigma) for y in Y, x in X]
  f /= sum(f)
  return f::Array{Float64,2}
end

# Create a binomial filter
function makebinomialfilter(size::Array{Int,2})
  # define combinatorics
  combi(n,k) = factorial(n)/factorial(k)/factorial(n-k)
  # define range in each dimension
  X =  [combi(size[2]-1,k) for k in 0:size[2]-1]'
  Y =  [combi(size[1]-1,k) for k in 0:size[1]-1]
  # generate binomial filter
  f = Y*X
  f /= sum(f)
  return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})
  D = A[1:2:end, 1:2:end]
  return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})
  B = zeros(size(A).*2)
  B[1:2:end, 1:2:end] = A
  U = 4*imfilter(B,makebinomialfilter(fsize),"symmetric")
  return U::Array{Float64,2}
end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)
  G = Array{Array{Float64,2}}(undef, nlevels)
  G[1] = im
  #G[1] = [convert(Array{Float64,2},im)]
  for n in 2:nlevels
    smooth = imfilter(G[n-1],makegaussianfilter(fsize,sigma),"symmetric")
    G[n] = downsample2(smooth)
  end
  return G::Array{Array{Float64,2},1}
end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})
  # define normalize function
  normal(A) = (A.-minimum(A)) / (maximum(A)-minimum(A))
  display = normal(P[1])
  for i in 2:size(P,1)
    Pi = padarray(normal(P[i]),Fill(0,(0,0),(size(P[1],1)-size(P[i],1),0)))
    display = hcat(display,Pi)
  end
  figure()
  imshow(display,"gray")
  return gcf()
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})
  # upsample Gaussian pyramid
  U = [upsample2(g,fsize) for g in G[2:nlevels]]
  # generate Laplacian pyramid
  L = G[:]
  L[1:end-1] -= U
  return L::Array{Array{Float64,2},1}
end

# Amplify frequencies of the first two layers of the laplacian pyramid
function amplifyhighfreq2(L::Array{Array{Float64,2},1})
  A = L[:]
  A[1] *= 1.5
  A[2] *= 1.5
  return A::Array{Array{Float64,2},1}
end

# Reconstruct an image from the laplacian pyramid
function reconstructlaplacianpyramid(L::Array{Array{Float64,2},1},fsize::Array{Int,2})
  im = L[end]
  for i in size(L,1)-1:-1:1
    im = upsample2(im,fsize) + L[i]
  end
  return im::Array{Float64,2}
end


# Problem 1: Image Pyramids and Image Sharpening

function problem1()
  # parameters
  fsize = [5 5]
  sigma = 1.4
  nlevels = 6

  # load image
  img = PyPlot.imread("a2p1.png")

  # create gaussian pyramid
  G = makegaussianpyramid(img,nlevels,fsize,sigma)

  # display gaussianpyramid
  displaypyramid(G)
  title("Gaussian Pyramid")

  # create laplacian pyramid
  L = makelaplacianpyramid(G,nlevels,fsize)

  # dispaly laplacian pyramid
  displaypyramid(L)
  title("Laplacian Pyramid")

  # amplify finest 2 subands
  L_amp = amplifyhighfreq2(L)

  # reconstruct image from laplacian pyramid
  im_rec = reconstructlaplacianpyramid(L_amp,fsize)

  # display original and reconstructed image
  figure()
  subplot(131)
  imshow(img,"gray",interpolation="none")
  axis("off")
  title("Original Image")
  subplot(132)
  imshow(im_rec,"gray",interpolation="none")
  axis("off")
  title("Reconstructed Image")
  subplot(133)
  imshow(img-im_rec,"gray",interpolation="none")
  axis("off")
  title("Difference")
  gcf()

  return
end
