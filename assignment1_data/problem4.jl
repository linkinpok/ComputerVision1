using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction
function createfilters()
  # derivative in x-direction using correlation
  deriv_x = [-0.5 0 0.5]
  # define gaussian function
  #gaussian(x,y,s)= 1/(2*pi*s*s) * exp(-(x*x+y*y)/(2*s*s))
  gaussian(x,s)= 1/sqrt(2*pi*s*s) * exp(-(x*x)/(2*s*s))
  # smooth in y-direction
  gauss_y = gaussian.([-1; 0; 1], 0.9)
  gauss_y /= norm(gauss_y,1)    # normalize
  # 3x3 filter
  fx = gauss_y * deriv_x
  fy = deriv_x' * gauss_y'  # here should be fy = fx', but it yields strange type: Adjoint{Float64,Array{Float64,2}}
  return fx::Array{Float64,2}, fy::Array{Float64,2}
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})
  Ix = imfilter(I,fx,"replicate")
  Iy = imfilter(I,fy,"replicate")
  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end


# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2},Iy::Array{Float64,2}, thr::Float64)
  edges = sqrt.(Ix.^2 + Iy.^2)
  edges .*= (edges .>= thr)     # only keep the values above the threshold
  return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})
  #direction = atan.(Iy,Ix)    # from -pi to pi (4 quadrants)
  direction = atan.(Iy./Ix)   # from -pi/2 to pi/2 (2 quadrants) => less cases

  # there are 4 possible directions of nearest neighbors
  x_downright = (-3pi/8 .<= direction .< -pi/8)
  x_right = (-pi/8 .<= direction .< pi/8)
  x_upright = (pi/8 .<= direction .< 3pi/8)
  x_up = (3pi/8 .<= direction) .| (direction .<= -3pi/8)

  # prepare a new matrix with zero padding in boundaries, index starts with 0!
  # Source: https://juliaimages.github.io/latest/function_reference.html#ImageFiltering.padarray
  padded = padarray(edges, Fill(0,(1,1),(1,1)))
  rows = 1 : size(edges,1)
  cols = 1 : size(edges,2)

  # find the pixels, which are not the maximum for each direction
  nonmax_downright = padded[rows,cols] .< max.(padded[rows.+1,cols.-1],padded[rows.-1,cols.+1])
  nonmax_right = padded[rows,cols] .< max.(padded[rows,cols.-1],padded[rows,cols.+1])
  nonmax_upright = padded[rows,cols] .< max.(padded[rows.-1,cols.-1],padded[rows.+1,cols.+1])
  nonmax_up = padded[rows,cols] .< max.(padded[rows.+1,cols],padded[rows.-1,cols])

  # eliminate value of those pixels
  edges[x_downright .* nonmax_downright] .= 0
  edges[x_right .* nonmax_right] .= 0
  edges[x_upright .* nonmax_upright] .= 0
  edges[x_up .* nonmax_up] .= 0

  # set value of the rest pixels to 1
  edges[edges .> 0] .= 1
  return edges::Array{Float64,2}
end


#= Problem 4
Image Filtering and Edge Detection =#

function problem4()

  # load image
  img = PyPlot.imread("a1p4.png")

  # create filters
  fx, fy = createfilters()

  # filter image
  imgx, imgy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(imgx, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(imgy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt.(imgx.^2 + imgy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  threshold = 10. / 255.
  edges = detectedges(imgx,imgy,threshold)
  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,imgx,imgy)
  figure()
  imshow(edges2,"gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()
  return
end

#= We decided to reduce the threshold in order to gain more important image edges.
Threshold = 10 is an acceptable choice that detects much details, even the edges
reflected on the lake surface.
A smaller threshold yields more details, but most os them are noise and unnecessary.
=#
