cylinder_size = 0.02;
// Cylider specifications
radius = 0.5;
x_centers[] = {-Cos(Pi/6)*3*radius, 0, 0};   // Cylinder centers form equilat tri with 3R side
y_centers[] = {0, -1.5*radius, 1.5*radius};  // 3R apart

box_size = 0.5;
// Box specs; these are hardcoded [-6, 20] x [-6, 6]
ll = newp;
Point(ll) = {-6, -6, 0, box_size};

lr = newp;
Point(lr) = {20, -6, 0, box_size};

ur = newp;
Point(ur) = {20, 6, 0, box_size};

ul = newp;
Point(ul) = {-6, 6, 0, box_size};

// Mark the boundary as follows
//             4
// ul(x)----------------ur(x)
// |                     |
// | 1                   | 2
// |                     |
// ll(x)----------------lr(x)
//             3

ll_lr = newl;
Line(ll_lr) = {ll, lr};
Physical Line(3) = {ll_lr};

lr_ur = newl;
Line(lr_ur) = {lr, ur};
Physical Line(2) = {lr_ur};

ur_ul = newl;
Line(ur_ul) = {ur, ul};
Physical Line(4) = {ur_ul}; 

ul_ll = newl;
Line(ul_ll) = {ul, ll};
Physical Line(1) = {ul_ll}; 

// Outer box
outer = newll;
Line Loop(outer) = {ll_lr, lr_ur, ur_ul, ul_ll};

cylinders[] = {};
cylinder_surfaces[] = {};
// For the cylinders let's define an auxliary macro
// assuming center_x, center_y and radius have their values in the namesspace
// and there is a tag
Macro MakeCylinder
      p = newp; 
      Point(p) = {center_x, center_y, 0, cylinder_size};
      Point(p+1) = {center_x-radius, center_y, 0, cylinder_size};
      Point(p+2) = {center_x, center_y-radius, 0, cylinder_size};
      Point(p+3) = {center_x+radius, center_y, 0, cylinder_size};
      Point(p+4) = {center_x, center_y+radius, 0, cylinder_size};

      l = newl;
      Circle(l) = {p+1, p, p+2};
      Circle(l+1) = {p+2, p, p+3};
      Circle(l+2) = {p+3, p, p+4};
      Circle(l+3) = {p+4, p, p+1};

      // All these take the same tag
      tag += 1;
      cylinder_surface[] = {l, l+1, l+2, l+3};
      Physical Line(tag) = {cylinder_surface[]};

      loop = newll;
      Line Loop(loop) = {cylinder_surface[]};
      cylinders[] += {loop};
      cylinder_surfaces[] += {cylinder_surface[]};
Return

tag = 4;  // The last tag used for surface marking
// Let there be cylinder
For i In {1:#x_centers[]}
    center_x = x_centers[i-1];
    center_y = y_centers[i-1];

    Call MakeCylinder;
EndFor

bounding_loops[] = {outer};
bounding_loops[] += {cylinders[]};
// Surface now is
s = news;
Plane Surface(s) = {bounding_loops[]};
Physical Surface(1) = {s};

Macro Extrema
  min_item = items[0];
  max_item = min_item;
  For i In {1:#items[]}
    thing = items[i-1];
    If(thing < min_item)
      min_item = thing;
    EndIf

    If(thing > max_item)
      max_item = thing;
    EndIf
  EndFor
Return


// For better control let us refine a bounding stripe of the cylinders
// Compute bounding box of the cylinders;
items[] = x_centers;
Call Extrema;
x_min = min_item - 2*radius;
x_max = max_item + 22*radius;  // Make it longer

items[] = y_centers;
Call Extrema;
y_min = min_item - 8*radius;
y_max = max_item + 8*radius;

// Mesh size for cells with XMin < x < Xmax and Ymin < y Ymax is set to
// VIn, outside we have Vout. Another possibility is to control the size
// by characteristic mesh size, i.e. the final argument of point constructor
Field[1] = Box;
Field[1].XMin = x_min;
Field[1].XMax = x_max;
Field[1].YMin = y_min; 
Field[1].YMax = y_max;
Field[1].VIn = 0.18; //2*cylinder_size;
Field[1].VOut = box_size;

// Alternative; fine inside ball
// Search for center
xcenter = 0;
ycenter = 0;
For i In {1:#x_centers[]}
    xcenter += x_centers[i-1];
    ycenter += y_centers[i-1];
EndFor

xcenter /= #x_centers[];
ycenter /= #x_centers[];

x_shifts[] = {-radius, 0, radius, 0};
y_shifts[] = {0, -radius, 0, radius};
// Search for ball radius
ball_size = 0;
For i In {1:#x_centers[]}
    x0 = x_centers[i-1];
    y0 = y_centers[i-1];

    For j In {1:#x_shifts[]}
       x = x0 + x_shifts[j-1];
       y = y0 + y_shifts[j-1];

       dist = Sqrt((x-xcenter)*(x-xcenter) + (y-ycenter)*(y-ycenter));
       If(dist > ball_size)
         ball_size = dist;
       EndIf
    EndFor
EndFor

Field[2] = Ball;
Field[2].XCenter = xcenter;
Field[2].YCenter = ycenter;
Field[2].Radius = ball_size + 0.5*radius;
Field[2].VIn = cylinder_size;
Field[2].VOut = box_size;

tag = 2;
ncyls = #x_centers[];
threshold_tags[] = {};
// Compute mesh size based on distance from cylinders
For c In {0:(ncyls-1)}
    c_surface[] = {};
    For i In {(c*4):((c+1)*4-1)}
        c_surface[] += {cylinder_surfaces[i]};
    EndFor

    tag += 1;
    // Setup distance field
    Field[tag] = Distance;
    Field[tag].EdgesList = {c_surface[]};
    // And threshold based on it
    ttag = tag + 1;
    Field[ttag] = Threshold;
    Field[ttag].IField = tag;
    Field[ttag].LcMin = cylinder_size;  // When dist < DistMin we have LcMin
    Field[ttag].LcMax = box_size;  
    Field[ttag].DistMin = 0.1*radius;
    Field[ttag].DistMax = 0.2*radius;
    
    threshold_tags[] += {ttag};
    tag += 1;
EndFor

tag += 1;
// It reamains to have master for the slave thresholds
Field[tag] = Min;
Field[tag].FieldsList = {threshold_tags[]};

// Which if any field to use for mesh size. Commenting out line below
// makes the mesh size depend only on size of points
Background Field = 1;
