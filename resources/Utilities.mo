within TypicalScensrio;

package Utilities "通用模型"


  package Functions
    annotation (Protection(access = Access.icon));
    function Map_VFpi "插值函数，先VF插值后PR插值"
      input Integer VF_n "体积流量个数";
      input Integer PR_n "压比个数";
      input Real Table[VF_n] "体积流量取值";
      input Real VF_PR_E[VF_n,PR_n,2] "所有体积流量对应的PR_E";
      input Real VF_in;
      input Real PR_in "压比";
      output Real Eta "效率";
    protected
      Real PR_E[PR_n,2] "转速输入后，插值得到的当前体积流量下的PR_E";
      Real PR_map "通用特性曲线下的压比";
      annotation (Protection(access = Access.icon));
    algorithm
      for i in 1:VF_n - 1 loop
        if VF_in >= Table[VF_n] then
          PR_E := VF_PR_E[VF_n,:,:];
        elseif VF_in < Table[1] then
          PR_E := VF_PR_E[1,:,:];
        elseif VF_in >= Table[i] and VF_in < Table[i + 1] then
          PR_E := VF_PR_E[i,:,:] + (VF_PR_E[i + 1,:,:] - VF_PR_E[i,:,:]) * (VF_in - Table[i]) / (Table[i + 1] - Table[i]);
        end if;
      end for;
      for i in 1:PR_n - 1 loop
        if PR_in >= PR_E[PR_n,1] then
          Eta := PR_E[PR_n,2];
        elseif PR_in < PR_E[1,1] then
          Eta := PR_E[1,2];
        elseif PR_in >= PR_E[i,1] and PR_in < PR_E[i + 1,1] then
          Eta := PR_E[i,2] + (PR_E[i + 1,2] - PR_E[i,2]) * (PR_in - PR_E[i,1]) / (PR_E[i + 1,1] - PR_E[i,1]);
        end if;
      end for;
    end Map_VFpi;
    function SplineInterpolation "Spline interpolation"
      input Real TabX[:] "References table";
      input Real TabY[:] "Results table";
      input Real X "Reference value"; // index in table
      input Real t(max = 1) = 0.5 "Stiffness parameter";
      output Real Y "Interpolated result";
    protected
      parameter Integer dimX = size(TabX, 1) "TabX dimension";
      parameter Integer dimY = size(TabY, 1) "TabY dimension";
      Real y0;
      Real y1;
      Real y2;
      Real x0;
      Real x1;
      Real x2;
      Real y1d "derivative at point at y1";
      Real y2d "derivative at point at y2";
      Integer IndX "Reference index";
      Boolean IndXcal "True if IndX is contained in TabX";
    algorithm
      if (dimX <> dimY) then
        assert(false, "LinearInterpolation: the dimensions of the tables are different");
      end if;
      if (dimX < 2) then
        assert(false, "Only one data point in table");
      end if;
      IndXcal := false;

      // Find index in table:
      if (X > TabX[1]) then
        for i in 2:dimX loop
          if ((X <= TabX[i]) and (not IndXcal)) then // IndX => x2
            IndX := i;
            IndXcal := true;
          end if;
        end for;
      end if;
      // If index is outside of table => Linear extrapolation
      if (not IndXcal) then
        if (X <= 2) then
          IndX := 2;
        else
          IndX := dimX;
        end if;
      end if;
      // Relevant data points:
      y1 := TabY[IndX - 1];
      y2 := TabY[IndX];
      x1 := TabX[IndX - 1];
      x2 := TabX[IndX];
      y2d := (y2 - y1) / (x2 - x1);// Approximating derivative

      // Use spline interpolation if X i contained in the table interval,
      //  If NOT contained: Use linear extrapolation.
      if (not IndXcal) then
        // Linear Extrapolation
        if (IndX == dimX) then
          Y := y2d * (X - x2) + y2;
        else
          Y := y2d * (X - x1) + y1;
        end if;
      else
        // Spline interpolation:
        if (IndX == 2) then
          y1d := y2d; // Approximating derivative
        else
          // In table (genral case)
          y0 := TabY[IndX - 2];
          x0 := TabX[IndX - 2];
          y1d := (1 - t) * 0.5 * (y2d + (y1 - y0) / (x1 - x0));// Approximating derivative
          if (IndX < dimX) then
            y2d := (1 - t) * 0.5 * (y2d + (TabY[IndX + 1] - y2) / (TabX[IndX + 1] - x2));
          end if;
        end if;
        // Compute Y using CubicHermite, spline interpolation
        Y := Utilities.Functions.CubicHermite(
          x = X,
          x1 = x1,
          x2 = x2,
          y1 = y1,
          y2 = y2,
          y1d = y1d,
          y2d = y2d);
      end if;
      annotation (
        smoothOrder = 1,
        Icon(graphics), Documentation(info = "<html>
<p><b>ThermoSysPro Version 3.1</b> </p>
<p>Spline interpolation function. The resulting spline will be continuous and have a continuous first derivative.</p>
<p><h4><font color=\"#008000\">Implementation</font></h4></p>
<p>It uses a cardinal spline interpolation algorithm. Cardinal splines are a sub-set of cubic Hermite splines where each piece is a third-degree polynomial specified in Hermite form: i.e specified by its values and the first derivatives at the end points of the reference interval.</p>
<p>The derivatives are calculated based on the non-uniform cardinal grid approach according to:</p>
<p>y1_der=0.5(1-t)(y1-y0/(x1-x0)+(y2-y1/(x2-x1)</p>
<p>I.e. the derivative in a point is calculated as an average of the surrounding points with an extra input shape parameter t.</p>
<p><h4><font color=\"#008000\">Inputs</font></h4></p>
<p><ul>
<li>TabX: Vector containing x-table values</li>
<li>TabY: Vector containing y-table values</li>
<li>X: The x-value that the spline should be evaluated at</li>
<li>t: Cardinal spline shape parameter. t = 0.5 is default and is generally a good choice. A value close to 1 will yield a stiff spline, t=0<a name=\"_x0000_i1025\">&nbsp;</a>corresponds to a Catmull-Rom spline and <a name=\"_x0000_i1025\">&nbsp;</a>t &LT; 0 corresponds to a more &ldquo;loose&rdquo; spline. From testing, t=0.5 <a name=\"_x0000_i1025\">&nbsp;</a>seems to be a good choice in general and is therefore chosen as a default value. A value of t = 1 corresponds to that the derivative in all data points is zero, which may result in strange curves . See Examples &GT; TestSplineInterpolation for a demonstration.</li>
</ul></p>
<p><h4><font color=\"#008000\">Output</font></h4></p>
<p><ul>
<li>Y: Interpolated value evaluated at X</li>
</ul></p>
<p><h4><font color=\"#008000\">Extrapolation</font></h4></p>
<p>Linear extrapolation is employed if the reference value is not contained in the reference value table.</p>
<p><h4><font color=\"#008000\">Example</font></h4></p>
<p><ul>
<li>TabX = {1,2,3,4};</li>
<li>TabY = {4,3,5,2};<br/><br/><img src=\"modelica://ThermoSysPro/Resources/Images/spline.png\"/></li>
</ul></p>
<p><h4><font color=\"#008000\">References</font></h4></p>
<p><a href=\"http://people.cs.clemson.edu/~dhouse/courses/405/notes/splines.pdf\">http://people.cs.clemson.edu/~dhouse/courses/405/notes/splines.pdf</a></p>
</html>", revisions = "<html>
</html>"), Protection(access = Access.icon));
    end SplineInterpolation;
    function CubicHermite "Evaluate a cubic Hermite spline"
      input Real x "Abscissa value";
      input Real x1 "Lower abscissa value";
      input Real x2 "Upper abscissa value";
      input Real y1 "Lower ordinate value";
      input Real y2 "Upper ordinate value";
      input Real y1d "Lower gradient";
      input Real y2d "Upper gradient";
      output Real y "Interpolated ordinate value";
    protected
      Real h "Distance between x1 and x2";
      Real t "abscissa scaled with h, i.e., t=[0..1] within x=[x1..x2]";
      Real h00 "Basis function 00 of cubic Hermite spline";
      Real h10 "Basis function 10 of cubic Hermite spline";
      Real h01 "Basis function 01 of cubic Hermite spline";
      Real h11 "Basis function 11 of cubic Hermite spline";
      Real aux3 "t cube";
      Real aux2 "t square";
    algorithm
      h := x2 - x1;
      if abs(h) > 0 then
        // Regular case
        t := (x - x1) / h;

        aux3 := t ^ 3;
        aux2 := t ^ 2;

        h00 := 2 * aux3 - 3 * aux2 + 1;
        h10 := aux3 - 2 * aux2 + t;
        h01 := -2 * aux3 + 3 * aux2;
        h11 := aux3 - aux2;
        y := y1 * h00 + h * y1d * h10 + y2 * h01 + h * y2d * h11;
      else
        // Degenerate case, x1 and x2 are identical, return step function
        y := (y1 + y2) / 2;
      end if;
      annotation (smoothOrder = 3, Documentation(revisions = "<html>
<p><u><b>Author</b></u></p>
<ul>
<li><i>May 2008</i> by <a href=\"mailto:Michael.Sielemann@dlr.de\">Michael Sielemann</a></li>
</ul>
</html>", info = "<html>
<p><b>ThermoSysPro Version 3.1</h4>
</html>"), Protection(access = Access.icon));
    end CubicHermite;
  end Functions;

  package Types
    type Cost = Real(final quantity = "Cost", final unit = "1") "成本" annotation (Protection(access = Access.icon));
    type ExergyCost = Real(final quantity = "1/Exergy", final unit = "1/J") "㶲成本" annotation (Protection(access = Access.icon));
    type ExergyCost_kWh = Real(final quantity = "1/Exergy", final unit = "1/kWh") "㶲成本" annotation (Protection(access = Access.icon));
    type CostFlow = Real(final quantity = "1/Modelica.SIunits.Time", final unit = "1/s") "成本流" annotation (Protection(access = Access.icon));
    type CostFlow_h = Real(final quantity = "1/Modelica.SIunits.Time", final unit = "1/h") annotation (Protection(access = Access.icon));
    annotation (Protection(access = Access.icon));
  end Types;

  package Blocks
    model Controller2 "分程控制"
      parameter Modelica.SIunits.Pressure P_k1 = 50;
      parameter Modelica.SIunits.Pressure P_k2 = 43;
      Modelica.Blocks.Interfaces.RealInput P 
        annotation (Placement(transformation(origin = {-100.0, 50.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput u 
        annotation (Placement(transformation(origin = {-100.0, -30.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealOutput y1 "主调节阀开度" 
        annotation (Placement(transformation(origin = {100.0, 60.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealOutput y2 "补气阀1开度" 
        annotation (Placement(transformation(origin = {100.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealOutput y3 "补气阀2开度" 
        annotation (Placement(transformation(origin = {100.0, -60.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Rectangle(origin = {-3.0, 3.0},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-93.0, 93.0}, {93.0, -93.0}}), Text(origin = {-3.0, 10.0},
        lineColor = {0, 0, 0},
        extent = {{-57.0, 24.0}, {57.0, -24.0}},
        textString = "分程控制",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None)}), Protection(access = Access.icon));
    equation
      if P > P_k1 then
        y1 = u;
        y2 = 0.001;
        y3 = 0.001;
      elseif P > P_k2 then
        y1 = 1;
        y2 = u;
        y3 = 0.001;
      else
        y1 = 1;
        y2 = 1;
        y3 = u;
      end if;
    end Controller2;
    annotation (Protection(access = Access.icon));

    model Control_Massflow
      parameter Modelica.SIunits.Pressure p_max = 1.08e7 "最高压力";
      parameter Modelica.SIunits.Pressure p_min = 5.739999999999999e6 "最低压力";
      parameter Modelica.SIunits.Time t = 3600 * 4 "静置时间";
      parameter Modelica.SIunits.MassFlowRate m_in = 237 "充气质量流量";
      parameter Modelica.SIunits.MassFlowRate m_out = -1 "放气质量流量";
      parameter Real table_charge[:,:] = {{5739000, 38.3, 1}, {10810000, 38.3, 2}};
      parameter Real table_discharge[:,:] = {{5739000, -27.8, 3}, {6830000, -25.1, 4}, {7930000, -24, 5}, {10810000, -24, 6}};

      // SI.Power P_charge;
      // SI.Power P_discharge;

      Modelica.Blocks.Interfaces.RealInput u 
        annotation (Placement(transformation(origin = {-117.99999999999997, -1.7763568394002505e-15},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealOutput y 
        annotation (Placement(transformation(origin = {110.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      inner Modelica.StateGraph.StateGraphRoot stateGraphRoot 
        annotation (Placement(transformation(origin = {-84.0, 60.00000000000001},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.StateGraph.InitialStep s0 annotation (Placement(transformation(origin = {-70, 10},
        extent = {{-10, -10}, {10, 10}})));



      Modelica.StateGraph.Step s1 annotation (Placement(transformation(origin = {-12, 10},
        extent = {{-10, -10}, {10, 10}})));
      Modelica.StateGraph.Transition transition2(enableTimer = false, waitTime = 1, condition = u >= p_max) 
        annotation (Placement(transformation(origin = {18, 10},
          extent = {{-10, -10}, {10, 10}})));
      Modelica.StateGraph.Step s2 annotation (Placement(transformation(origin = {48, 10},
        extent = {{-10, -10}, {10, 10}})));
      Modelica.StateGraph.Transition transition3(enableTimer = true, waitTime = t, condition = true) 
        annotation (Placement(transformation(origin = {78, 10},
          extent = {{-10, -10}, {10, 10}})));
      Modelica.StateGraph.Step s3 annotation (Placement(transformation(origin = {48, -46},
        extent = {{10, -10}, {-10, 10}})));
      Modelica.StateGraph.Transition transition4(enableTimer = false, waitTime = 1, condition = u <= p_min) 
        annotation (Placement(transformation(origin = {18, -46},
          extent = {{10, -10}, {-10, 10}})));
      Modelica.StateGraph.Step s4 annotation (Placement(transformation(origin = {-12, -46},
        extent = {{10, -10}, {-10, 10}})));
      Modelica.StateGraph.Transition transition5(enableTimer = true, waitTime = t, condition = true) 
        annotation (Placement(transformation(origin = {-42, -46},
          extent = {{10, -10}, {-10, 10}})));
      annotation (Diagram(coordinateSystem(extent = {{-100, -100}, {100, 100}},
        grid = {2, 2})),
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, 0.0},
          extent = {{-100.0, 100.0}, {100.0, -100.0}}), Rectangle(origin = {-63.0, -3.0000000000000036},
          extent = {{-19.0, 17.0}, {19.0, -17.0}}), Line(origin = {0.0, -5.0000000000000036},
          points = {{0.0, 35.0}, {0.0, -35.0}}), Rectangle(origin = {67.0, -3.0000000000000036},
          extent = {{-19.0, 17.0}, {19.0, -17.0}}), Polygon(origin = {-6.0, -2.0000000000000036},
          fillPattern = FillPattern.Solid,
          points = {{-6.0, 6.0}, {6.0, 0.0}, {-6.0, -6.0}, {-6.0, 6.0}}), Line(origin = {-28.0, -2.0000000000000036},
          points = {{-16.0, 0.0}, {16.0, 0.0}}), Polygon(origin = {42.0, -2.0000000000000036},
          fillPattern = FillPattern.Solid,
          points = {{-6.0, 6.0}, {6.0, 0.0}, {-6.0, -6.0}, {-6.0, 6.0}}), Line(origin = {18.0, -2.0000000000000036},
          points = {{-18.0, 0.0}, {18.0, 0.0}})}));
      Modelica.StateGraph.Transition transition6(enableTimer = false, waitTime = t, condition = true) 
        annotation (Placement(transformation(origin = {-41.0, 9.999999999999996},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Tables.CombiTable1Ds mflow_in(table = table_charge) 
        annotation (Placement(transformation(origin = {-8.0, 40.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Tables.CombiTable1Ds mflow_out(table = table_discharge) 
        annotation (Placement(transformation(origin = {-12.0, -74.00000000000003},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealOutput P_charge 
        annotation (Placement(transformation(origin = {110.00000000000003, -42.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealOutput P_discharge 
        annotation (Placement(transformation(origin = {110.0, 34.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    equation
      u = mflow_in.u;
      u = mflow_out.u;


      if s1.active then
        // y = m_in;
        y = mflow_in.y[1];
        P_charge = mflow_in.y[2];
        P_discharge = 0;
      else
        if s3.active then
          // y = m_out;
          y = mflow_out.y[1];
          P_charge = 0;
          P_discharge = mflow_out.y[2];
        else
          y = 0;
          P_charge = 0;
          P_discharge = 0;
        end if;
      end if;
      connect(s0.outPort[1], transition6.inPort) 
        annotation (Line(origin = {-53.0, 10.0},
          points = {{-6.5, 0.0}, {8.0, -3.552713678800501e-15}},
          color = {0, 0, 0}));

      connect(s1.outPort[1], transition2.inPort) 
        annotation (Line(origin = {6.0, 10.0},
          points = {{-7.5, 0.0}, {8.0, 0.0}},
          color = {0, 0, 0}));
      connect(transition2.outPort, s2.inPort[1]) 
        annotation (Line(origin = {28.0, 10.0},
          points = {{-8.5, 0.0}, {9.0, 0.0}},
          color = {0, 0, 0}));
      connect(s2.outPort[1], transition3.inPort) 
        annotation (Line(origin = {66.0, 10.0},
          points = {{-7.5, 0.0}, {8.0, 0.0}},
          color = {0, 0, 0}));
      connect(transition3.outPort, s3.inPort[1]) 
        annotation (Line(origin = {90.0, -18.0},
          points = {{-10.5, 28.0}, {6.0, 28.0}, {6.0, -28.0}, {-31.0, -28.0}},
          color = {0, 0, 0}));
      connect(s3.outPort[1], transition4.inPort) 
        annotation (Line(origin = {30.0, -46.0},
          points = {{7.5, 0.0}, {-8.0, 0.0}},
          color = {0, 0, 0}));
      connect(transition4.outPort, s4.inPort[1]) 
        annotation (Line(origin = {8.0, -46.0},
          points = {{8.5, 0.0}, {-9.0, 0.0}},
          color = {0, 0, 0}));
      connect(s4.outPort[1], transition5.inPort) 
        annotation (Line(origin = {-30.0, -46.0},
          points = {{7.5, 0.0}, {-8.0, 0.0}},
          color = {0, 0, 0}));
      connect(transition5.outPort, s0.inPort[1]) 
        annotation (Line(origin = {-80.0, -18.0},
          points = {{36.5, -28.0}, {-10.0, -28.0}, {-10.0, 28.0}, {-1.0, 28.0}},
          color = {0, 0, 0}));
      connect(transition6.outPort, s1.inPort[1]) 
        annotation (Line(origin = {-31.0, 10.0},
          points = {{-8.5, -3.552713678800501e-15}, {8.0, 0.0}},
          color = {0, 0, 0}));
    end Control_Massflow;
    model Controller "分程控制"
      // Modelica.Blocks.Interfaces.RealInput P
      //   annotation (Placement(transformation(origin = {-100.0, 50.0}, 
      //     extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      parameter Real table[:,:] = {{0, 0, 0, 0}, {0.333, 1, 0, 0}, {0.666, 1, 1, 0}, {1, 1, 1, 1}};
      Modelica.Blocks.Interfaces.RealInput u 
        annotation (Placement(transformation(origin = {-100.0, -30.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealOutput y1 "主调节阀开度" 
        annotation (Placement(transformation(origin = {100.0, 60.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealOutput y2 "补气阀1开度" 
        annotation (Placement(transformation(origin = {100.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealOutput y3 "补气阀2开度" 
        annotation (Placement(transformation(origin = {100.0, -60.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Rectangle(origin = {-3.0, 3.0},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-93.0, 93.0}, {93.0, -93.0}}), Text(origin = {-3.0, 10.0},
        lineColor = {0, 0, 0},
        extent = {{-57.0, 24.0}, {57.0, -24.0}},
        textString = "分程控制",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None)}), Protection(access = Access.icon));
      Modelica.Blocks.Tables.CombiTable1Ds Table(table = table,
        extrapolation = Modelica.Blocks.Types.Extrapolation.NoExtrapolation) annotation (Placement(transformation(origin = {-10.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    equation
      u = Table.u;
      y1 = Table.y[1];
      y2 = Table.y[2];
      y3 = Table.y[3];

      // if P > 8.459e6 then 
      //   y1 = u;
      //   y2 = 0.001;
      //   y3 = 0.001;
      // elseif P > 7.28e6 then 
      //   y1 = 1;
      //   y2 = u;
      //   y3 = 0.001;
      // else
      //   y1 = 1;
      //   y2 = 1;
      //   y3 = u;
      // end if;
    end Controller;
    model CompressorSwitch "压缩机开关"
      replaceable package Medium = Media.FluidMedia.IdealAir 
         constrainedby TypicalScensrio.Media.FluidMedia.PartialMedium 
          annotation (Dialog(tab = "工质", group = "工质选择"), choicesAllMatching = true, Protection(access = Access.diagram));
      parameter Modelica.SIunits.Pressure p_on = 58.9e5 "末段压缩机开启点压力" 
        annotation (Dialog(tab = "高级"));

      Interfaces.Fluid.FluidPort_a port_a(redeclare package Medium = Medium) "进口" annotation (
        Placement(transformation(origin = {-100.0, -4.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Interfaces.Fluid.FluidPort_b port_b(redeclare package Medium = Medium) "出口" annotation (
        Placement(transformation(origin = {100.0, 48.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Interfaces.Fluid.FluidPort_b port_c(redeclare package Medium = Medium) "出口" annotation (
        Placement(transformation(origin = {100.0, -44.00000000000001},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    equation
      if port_b.p > p_on then
      else
          // pi_C = 1;
          // phic = Phic(beta, N_T)
        Phic.u1 = N_norm;
        Phic.u2 = 0;
        W_norm = Phic.y;

          // eta = Eta(beta, N_T)
        Eta.u1 = N_norm;
        Eta.u2 = 0;
        eta_norm = Eta.y;

          // PR = PressRatio(beta, N_T)
        PressRatio.u1 = N_norm;
        PressRatio.u2 = 0;
        pi_norm = PressRatio.y;
      end if;
    end CompressorSwitch;

    model Controller1 "压缩机开关"
      parameter Modelica.SIunits.Pressure p_on = 5.889999999999999e6;
      // parameter SI.Pressure P_k2 = 43;
      Modelica.Blocks.Interfaces.RealInput P 
        annotation (Placement(transformation(origin = {-100.0, 50.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput u 
        annotation (Placement(transformation(origin = {-100.0, -30.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealOutput y1 
        annotation (Placement(transformation(origin = {98.0, 50.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealOutput y2 
        annotation (Placement(transformation(origin = {98.0, -10.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));



      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Rectangle(origin = {-3.0, 3.0},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-93.0, 93.0}, {93.0, -93.0}}), Text(origin = {-3.0, 10.0},
        lineColor = {0, 0, 0},
        extent = {{-57.0, 24.0}, {57.0, -24.0}},
        textString = "分程控制",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None)}), Protection(access = Access.icon));
    equation
      if P > p_on then
        y1 = u * 885.196090026484;
        y2 = 0;
      else
        y1 = 0;
        y2 = u;
      end if;
    end Controller1;

    model StateSwitching "状态切换"
      parameter Modelica.SIunits.Pressure p_on = 5.889999999999999e6;
      Modelica.SIunits.Pressure p_pre;
      Modelica.Blocks.Interfaces.RealInput p 
        annotation (Placement(transformation(origin = {-100.0, 50.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      // Modelica.Blocks.Interfaces.RealInput u
      //   annotation (Placement(transformation(origin = {-100.0, -30.0}, 
      //     extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealOutput y 
        annotation (Placement(transformation(origin = {98.0, 50.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      // Modelica.Blocks.Interfaces.RealOutput y2 "" 
      //   annotation (Placement(transformation(origin = {98.0, -10.0}, 
      //     extent = {{-10.0, -10.0}, {10.0, 10.0}})));



      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Rectangle(origin = {-3.0, 3.0},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-93.0, 93.0}, {93.0, -93.0}}), Text(origin = {-3.0, 10.0},
        lineColor = {0, 0, 0},
        extent = {{-57.0, 24.0}, {57.0, -24.0}},
        textString = "分程控制",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None)}), Protection(access = Access.icon));
    equation
      when sample(0, 0.1) then
        p_pre = pre(p);
      end when;
      if p > p_pre then
        if p >= p_on then
          y = p_on;
        else
          y = p;
        end if;
      else
        y = p_on;
      end if;
    end StateSwitching;
  end Blocks;

  annotation (Protection(access = Access.icon));
  package Interpolations "插值表"

    model Interpolation_3D
      parameter Integer nx1 = 3;
      parameter Integer nx2 = 4;
      parameter Integer nx3 = 5;
      parameter Real table[nx3,nx1 + 1,nx2 + 1] = {{{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}} "数据表，[x3,x1,x2]";
      parameter Real ux[nx3] = {1, 2, 3, 4, 5};

      Real uy[nx3];
      Modelica.Blocks.Interfaces.RealOutput y 
        annotation (Placement(transformation(origin = {110.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Tables.CombiTable2D IP_table[nx3](table = table) 
        annotation (Placement(transformation(origin = {-1.0000000000000018, 9.999999999999998},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput x1 
        annotation (Placement(transformation(origin = {-129.99999999999997, 64.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput x2 
        annotation (Placement(transformation(origin = {-130.0, 1.7763568394002505e-15},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput x3 
        annotation (Placement(transformation(origin = {-130.0, -64.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
    protected
      Integer index_1D;
      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Text(origin = {0.0, 130.0},
        lineColor = {0, 0, 255},
        extent = {{-150.0, 20.0}, {150.0, -20.0}},
        textString = "%name",
        textColor = {0, 0, 255}), Rectangle(origin = {0.0, 7.105427357601002e-15},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-100.0, -100.0}, {100.0, 100.0}}), Rectangle(origin = {13.000000000000075, 20.0},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-60.00000000000006, -40.0}, {59.99999999999994, 39.99999999999999}}), Rectangle(origin = {57.99999999999996, -10.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Line(origin = {13.000000000000018, 20.0},
        points = {{-60.0, 40.0}, {-60.0, -40.0}, {60.0, -40.0}, {60.0, 40.0}, {30.0, 40.0}, {30.0, -40.0}, {-30.0, -40.0}, {-30.0, 40.0}, {-60.0, 40.0}, {-60.0, 20.0}, {60.0, 20.0}, {60.0, 0.0}, {-60.0, 0.0}, {-60.0, -20.0}, {60.0, -20.0}, {60.0, -40.0}, {-60.0, -40.0}, {-60.0, 40.0}, {60.0, 40.0}, {60.0, -40.0}}), Line(origin = {13.000000000000018, 20.0},
        points = {{0.0, 40.0}, {0.0, -40.0}}), Line(origin = {13.000000000000018, 20.0},
        points = {{-60.0, -40.0}, {60.0, 40.0}},
        color = {0, 85, 255},
        thickness = 1.0,
        smooth = Smooth.Bezier), Rectangle(origin = {-0.9999999999999263, 0.0},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-60.00000000000006, -40.0}, {59.99999999999994, 39.99999999999999}}), Line(origin = {-0.9999999999999831, 0.0},
        points = {{-60.0, 40.0}, {-60.0, -40.0}, {60.0, -40.0}, {60.0, 40.0}, {30.0, 40.0}, {30.0, -40.0}, {-30.0, -40.0}, {-30.0, 40.0}, {-60.0, 40.0}, {-60.0, 20.0}, {60.0, 20.0}, {60.0, 0.0}, {-60.0, 0.0}, {-60.0, -20.0}, {60.0, -20.0}, {60.0, -40.0}, {-60.0, -40.0}, {-60.0, 40.0}, {60.0, 40.0}, {60.0, -40.0}}), Line(origin = {-0.9999999999999831, 0.0},
        points = {{0.0, 40.0}, {0.0, -40.0}}), Line(origin = {-0.9999999999999831, 0.0},
        points = {{-60.0, -40.0}, {60.0, 40.0}},
        color = {0, 128, 0},
        thickness = 1.0,
        smooth = Smooth.Bezier), Rectangle(origin = {44.000000000000014, -30.000000000000007},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-14.999999999999927, -20.0},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-60.00000000000006, -40.0}, {59.99999999999994, 39.99999999999999}}), Line(origin = {-14.999999999999984, -20.0},
        points = {{-60.0, 40.0}, {-60.0, -40.0}, {60.0, -40.0}, {60.0, 40.0}, {30.0, 40.0}, {30.0, -40.0}, {-30.0, -40.0}, {-30.0, 40.0}, {-60.0, 40.0}, {-60.0, 20.0}, {60.0, 20.0}, {60.0, 0.0}, {-60.0, 0.0}, {-60.0, -20.0}, {60.0, -20.0}, {60.0, -40.0}, {-60.0, -40.0}, {-60.0, 40.0}, {60.0, 40.0}, {60.0, -40.0}}), Line(origin = {-14.999999999999984, -20.0},
        points = {{0.0, 40.0}, {0.0, -40.0}}), Line(origin = {1.5000000000000018, -60.0},
        points = {{-77.5, 0.0}, {77.5, 0.0}},
        thickness = 1.5,
        arrow = {Arrow.None, Arrow.Filled},
        arrowSize = 10.0), Rectangle(origin = {-59.999999999999986, -50.000000000000014},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-59.999999999999986, -30.000000000000007},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-59.999999999999986, -10.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-59.999999999999986, 10.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Line(origin = {-14.999999999999984, -20.0},
        points = {{-60.0, -40.0}, {60.0, 40.0}},
        color = {255, 0, 0},
        thickness = 1.0,
        smooth = Smooth.Bezier), Rectangle(origin = {-29.999999999999982, -50.000000000000014},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {29.999999999999986, -50.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {1.687538997430238e-14, -50.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-45.999999999999986, 30.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-31.999999999999982, 50.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Line(origin = {-76.0, 4.000000000000057},
        points = {{0.0, -64.0}, {0.0, 64.0}},
        thickness = 1.5,
        arrow = {Arrow.None, Arrow.Filled},
        arrowSize = 10.0)}));
    equation
      for i in 1:nx3 loop
        IP_table[i].u1 = x1;
        IP_table[i].u2 = x2;
        IP_table[i].y = uy[i];
      end for;
      // algorithm 
      if x3 <= ux[1] then
        index_1D = 2;
        y = (uy[index_1D] - uy[index_1D - 1]) / (ux[index_1D] - ux[index_1D - 1]) * (x3 - ux[index_1D - 1]) + uy[index_1D - 1];
      elseif x3 > ux[end] then
        index_1D = nx3;
        y = (uy[index_1D] - uy[index_1D - 1]) / (ux[index_1D] - ux[index_1D - 1]) * (x3 - ux[index_1D - 1]) + uy[index_1D - 1];
      else
        index_1D = nx3;
        y = Modelica.Math.Vectors.interpolate(ux, uy, x3);
      end if;
    end Interpolation_3D;
    model Interpolation_3Ds
      parameter Integer nx1 = 3 "x1数量";
      parameter Integer nx2 = 4 "x2数量";
      parameter Integer nx3 = 5 "x3数量";

      parameter Integer ny = 3 "输出数量";
      parameter Real table_y1[nx3,nx1 + 1,nx2 + 1] = {{{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}} "数据表，[x3,x1,x2]";
      parameter Real table_y2[nx3,nx1 + 1,nx2 + 1] = {{{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}} "数据表，[x3,x1,x2]";
      parameter Real table_y3[nx3,nx1 + 1,nx2 + 1] = {{{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}, {{0, 1, 2, 3, 4}, {1, 1.5, 6.5, 11.5, 16.5}, {2, 2.5, 7.5, 12.5, 17.5}, {3, 3.5, 8.5, 13.5, 18.5}}} "数据表，[x3,x1,x2]";

      parameter Real ux[nx3] = {1, 2, 3, 4, 5} "x3实际值";

      Modelica.Blocks.Interfaces.RealOutput[ny] y 
        annotation (Placement(transformation(origin = {110.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));

      Modelica.Blocks.Interfaces.RealInput x1 
        annotation (Placement(transformation(origin = {-122.0, 52.4},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput x2 
        annotation (Placement(transformation(origin = {-121.99999999999999, -0.8000000000000025},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput x3 
        annotation (Placement(transformation(origin = {-122.00000000000001, -54.00000000000001},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Text(origin = {0.0, 130.0},
        lineColor = {0, 0, 255},
        extent = {{-150.0, 20.0}, {150.0, -20.0}},
        textString = "%name",
        textColor = {0, 0, 255}), Rectangle(origin = {0.0, 7.105427357601002e-15},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-100.0, -100.0}, {100.0, 100.0}}), Rectangle(origin = {13.000000000000075, 20.0},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-60.00000000000006, -40.0}, {59.99999999999994, 39.99999999999999}}), Rectangle(origin = {57.99999999999996, -10.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Line(origin = {13.000000000000018, 20.0},
        points = {{-60.0, 40.0}, {-60.0, -40.0}, {60.0, -40.0}, {60.0, 40.0}, {30.0, 40.0}, {30.0, -40.0}, {-30.0, -40.0}, {-30.0, 40.0}, {-60.0, 40.0}, {-60.0, 20.0}, {60.0, 20.0}, {60.0, 0.0}, {-60.0, 0.0}, {-60.0, -20.0}, {60.0, -20.0}, {60.0, -40.0}, {-60.0, -40.0}, {-60.0, 40.0}, {60.0, 40.0}, {60.0, -40.0}}), Line(origin = {13.000000000000018, 20.0},
        points = {{0.0, 40.0}, {0.0, -40.0}}), Line(origin = {13.000000000000018, 20.0},
        points = {{-60.0, -40.0}, {60.0, 40.0}},
        color = {0, 85, 255},
        thickness = 1.0,
        smooth = Smooth.Bezier), Rectangle(origin = {-0.9999999999999263, 0.0},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-60.00000000000006, -40.0}, {59.99999999999994, 39.99999999999999}}), Line(origin = {-0.9999999999999831, 0.0},
        points = {{-60.0, 40.0}, {-60.0, -40.0}, {60.0, -40.0}, {60.0, 40.0}, {30.0, 40.0}, {30.0, -40.0}, {-30.0, -40.0}, {-30.0, 40.0}, {-60.0, 40.0}, {-60.0, 20.0}, {60.0, 20.0}, {60.0, 0.0}, {-60.0, 0.0}, {-60.0, -20.0}, {60.0, -20.0}, {60.0, -40.0}, {-60.0, -40.0}, {-60.0, 40.0}, {60.0, 40.0}, {60.0, -40.0}}), Line(origin = {-0.9999999999999831, 0.0},
        points = {{0.0, 40.0}, {0.0, -40.0}}), Line(origin = {-0.9999999999999831, 0.0},
        points = {{-60.0, -40.0}, {60.0, 40.0}},
        color = {0, 128, 0},
        thickness = 1.0,
        smooth = Smooth.Bezier), Rectangle(origin = {44.000000000000014, -30.000000000000007},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-14.999999999999927, -20.0},
        lineColor = {0, 0, 127},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-60.00000000000006, -40.0}, {59.99999999999994, 39.99999999999999}}), Line(origin = {-14.999999999999984, -20.0},
        points = {{-60.0, 40.0}, {-60.0, -40.0}, {60.0, -40.0}, {60.0, 40.0}, {30.0, 40.0}, {30.0, -40.0}, {-30.0, -40.0}, {-30.0, 40.0}, {-60.0, 40.0}, {-60.0, 20.0}, {60.0, 20.0}, {60.0, 0.0}, {-60.0, 0.0}, {-60.0, -20.0}, {60.0, -20.0}, {60.0, -40.0}, {-60.0, -40.0}, {-60.0, 40.0}, {60.0, 40.0}, {60.0, -40.0}}), Line(origin = {-14.999999999999984, -20.0},
        points = {{0.0, 40.0}, {0.0, -40.0}}), Line(origin = {1.5000000000000018, -60.0},
        points = {{-77.5, 0.0}, {77.5, 0.0}},
        thickness = 1.5,
        arrow = {Arrow.None, Arrow.Filled},
        arrowSize = 10.0), Rectangle(origin = {-59.999999999999986, -50.000000000000014},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-59.999999999999986, -30.000000000000007},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-59.999999999999986, -10.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-59.999999999999986, 10.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Line(origin = {-14.999999999999984, -20.0},
        points = {{-60.0, -40.0}, {60.0, 40.0}},
        color = {255, 0, 0},
        thickness = 1.0,
        smooth = Smooth.Bezier), Rectangle(origin = {-29.999999999999982, -50.000000000000014},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {29.999999999999986, -50.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {1.687538997430238e-14, -50.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-45.999999999999986, 30.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Rectangle(origin = {-31.999999999999982, 50.0},
        fillColor = {255, 215, 136},
        fillPattern = FillPattern.Solid,
        extent = {{-15.0, -10.0}, {15.0, 10.0}}), Line(origin = {-76.0, 4.000000000000057},
        points = {{0.0, -64.0}, {0.0, 64.0}},
        thickness = 1.5,
        arrow = {Arrow.None, Arrow.Filled},
        arrowSize = 10.0)}));

      // Interpolation_3D[ny] interpolation_3D(nx1 = nx1, nx2 = nx2, nx3 = nx3, table = table, ux = ux)
      //   annotation (Placement(transformation(origin = {-0.9999999999999973, -0.8000000000000025}, 
      //     extent = {{-20.0, -20.0}, {20.0, 20.000000000000004}})));
      Interpolation_3D[ny] interpolation_3D(each nx1 = nx1, each nx2 = nx2, each nx3 = nx3, table = {table_y1, table_y2, table_y3}, each ux = ux) 
        annotation (Placement(transformation(origin = {-0.9999999999999973, -0.8000000000000025},
          extent = {{-20.0, -20.0}, {20.0, 20.000000000000004}})));
    equation
      for i in 1:ny loop

        interpolation_3D[i].x1 = x1;
        interpolation_3D[i].x2 = x2;
        interpolation_3D[i].x3 = x3;
        interpolation_3D[i].y = y[i];
      end for;
    end Interpolation_3Ds;
  end Interpolations;
  package TimeTable


    block TimeTable
      "Generate a (possibly discontinuous) signal by linear interpolation in a table"

      input Real table[:,2] = fill(0.0, 0, 2)
        "Table matrix (time = first column; e.g., table=[0, 0; 1, 1; 2, 4])" 
        annotation (Dialog(groupImage = "modelica://Modelica/Resources/Images/Blocks/Sources/TimeTable.png"));
      parameter Modelica.SIunits.Time timeScale(
        min = Modelica.Constants.eps) = 1 "Time scale of first table column" 
        annotation (Evaluate = true);
      extends Interfaces.SignalSource;
      parameter Modelica.SIunits.Time shiftTime = startTime
        "Shift time of first table column";
    protected
      Real a "Interpolation coefficient a of actual interval (y=a*x+b)";
      Real b "Interpolation coefficient b of actual interval (y=a*x+b)";
      Integer last(start = 1) "Last used lower grid index";
      discrete SIunits.Time nextEvent(start = 0, fixed = true) "Next event instant";
      discrete Real nextEventScaled(start = 0, fixed = true)
        "Next scaled event instant";
      Real timeScaled "Scaled time";

      function getInterpolationCoefficients
        "Determine interpolation coefficients and next time event"
        extends Modelica.Icons.Function;
        input Real table[:,2] "Table for interpolation";
        input Real offset "y-offset";
        input Real startTimeScaled "Scaled time-offset";
        input Real timeScaled "Actual scaled time instant";
        input Integer last "Last used lower grid index";
        input Real TimeEps "Relative epsilon to check for identical time instants";
        input Real shiftTimeScaled "Time shift";
        output Real a "Interpolation coefficient a (y=a*x + b)";
        output Real b "Interpolation coefficient b (y=a*x + b)";
        output Real nextEventScaled "Next scaled event instant";
        output Integer next "New lower grid index";
      protected
        Integer columns = 2 "Column to be interpolated";
        Integer ncol = 2 "Number of columns to be interpolated";
        Integer nrow = size(table, 1) "Number of table rows";
        Integer next0;
        Real tp;
        Real dt;
      algorithm
        next := last;
        nextEventScaled := timeScaled - TimeEps * abs(timeScaled);
        // in case there are no more time events
        tp := timeScaled + TimeEps * abs(timeScaled);

        if tp < startTimeScaled then
          // First event not yet reached
          nextEventScaled := startTimeScaled;
          a := 0;
          b := offset;
        elseif nrow < 2 then
          // Special action if table has only one row
          a := 0;
          b := offset + table[1,columns];
        else
          tp := tp - shiftTimeScaled;
          // Find next time event instant. Note, that two consecutive time instants
          // in the table may be identical due to a discontinuous point.
          while next < nrow and tp >= table[next,1] loop
            next := next + 1;
          end while;
          // Define next time event, if last table entry not reached
          if next < nrow then
            nextEventScaled := shiftTimeScaled + table[next,1];
          end if;
          // Determine interpolation coefficients
          if next == 1 then
            next := 2;
          end if;
          next0 := next - 1;
          dt := table[next,1] - table[next0,1];
          if dt <= TimeEps * abs(table[next,1]) then
            // Interpolation interval is not big enough, use "next" value
            a := 0;
            b := offset + table[next,columns];
          else
            a := (table[next,columns] - table[next0,columns]) / dt;
            b := offset + table[next0,columns] - a * table[next0,1];
          end if;
        end if;
        // Take into account shiftTimeScaled "a*(time - shiftTime) + b"
        b := b - a * shiftTimeScaled;
      end getInterpolationCoefficients;
      import Modelica.Blocks.Interfaces;
      import Modelica.SIunits;
    algorithm
      if noEvent(size(table, 1) > 1) then
        assert(not (table[1,1] > 0.0 or table[1,1] < 0.0), "The first point in time has to be set to 0, but is table[1,1] = " + String(table[1,1]));
      end if;
      when {time >= pre(nextEvent), initial()} then
        (a,b,nextEventScaled,last) := getInterpolationCoefficients(
          table,
          offset,
          startTime / timeScale,
          timeScaled,
          last,
          100 * Modelica.Constants.eps,
          shiftTime / timeScale);
        nextEvent := nextEventScaled * timeScale;
      end when;
    equation
      assert(size(table, 1) > 0, "No table values defined.");
      timeScaled = time / timeScale;
      y = a * timeScaled + b;
      annotation (
        Icon(coordinateSystem(
          preserveAspectRatio = true,
          extent = {{-100, -100}, {100, 100}}), graphics = {
          Line(points = {{-80, 68}, {-80, -80}}, color = {192, 192, 192}),
          Polygon(
          points = {{-80, 90}, {-88, 68}, {-72, 68}, {-80, 90}},
          lineColor = {192, 192, 192},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Solid),
          Line(points = {{-90, -70}, {82, -70}}, color = {192, 192, 192}),
          Polygon(
          points = {{90, -70}, {68, -62}, {68, -78}, {90, -70}},
          lineColor = {192, 192, 192},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Solid),
          Rectangle(
          extent = {{-48, 70}, {2, -50}},
          lineColor = {255, 255, 255},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Solid),
          Line(points = {{-48, -50}, {-48, 70}, {52, 70}, {52, -50}, {-48, -50}, {-48, -20},
          {52, -20}, {52, 10}, {-48, 10}, {-48, 40}, {52, 40}, {52, 70}, {2, 70}, {2, -51}}),
          Text(
          extent = {{-150, -150}, {150, -110}},
          textString = "offset=%offset")}),
        Diagram(coordinateSystem(
          preserveAspectRatio = true,
          extent = {{-100, -100}, {100, 100}}), graphics = {
          Polygon(
          points = {{-80, 90}, {-85, 68}, {-74, 68}, {-80, 90}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Line(points = {{-80, 68}, {-80, -80}}, color = {95, 95, 95}),
          Line(points = {{-90, -70}, {82, -70}}, color = {95, 95, 95}),
          Polygon(
          points = {{88, -70}, {68, -65}, {68, -74}, {88, -70}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Rectangle(
          extent = {{-20, 90}, {30, -30}},
          lineColor = {255, 255, 255},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Solid),
          Line(points = {{-20, -30}, {-20, 90}, {80, 90}, {80, -30}, {-20, -30}, {-20, 0}, {
          80, 0}, {80, 30}, {-20, 30}, {-20, 60}, {80, 60}, {80, 90}, {30, 90}, {30, -31}}),
          Text(
          extent = {{-70, -42}, {-32, -54}},
          textString = "offset"),
          Polygon(
          points = {{-31, -30}, {-33, -40}, {-28, -40}, {-31, -30}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Polygon(
          points = {{-31, -70}, {-34, -60}, {-29, -60}, {-31, -70}, {-31, -70}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Line(points = {{-31, -32}, {-31, -70}}, color = {95, 95, 95}),
          Line(points = {{-20, -30}, {-20, -70}}, color = {95, 95, 95}),
          Text(
          extent = {{-38, -73}, {8, -83}},
          textString = "startTime"),
          Line(points = {{-20, -30}, {-80, -30}}, color = {95, 95, 95}),
          Text(
          extent = {{-76, 93}, {-44, 75}},
          textString = "y"),
          Text(
          extent = {{66, -78}, {90, -88}},
          textString = "time"),
          Text(
          extent = {{-15, 83}, {24, 68}},
          textString = "time"),
          Text(
          extent = {{33, 83}, {76, 67}},
          textString = "y")}),
        Documentation(info = "<html>
<p>
This block generates an output signal by <strong>linear interpolation</strong> in
a table. The time points and function values are stored in a matrix
<strong>table[i,j]</strong>, where the first column table[:,1] contains the
time points and the second column contains the data to be interpolated.
The table interpolation has the following properties:
</p>
<ul>
<li>The interpolation interval is found by a linear search where the interval used in the
    last call is used as start interval.</li>
<li>The time points need to be <strong>monotonically increasing</strong>.</li>
<li><strong>Discontinuities</strong> are allowed, by providing the same
    time point twice in the table.</li>
<li>Values <strong>outside</strong> of the table range, are computed by
    <strong>extrapolation</strong> through the last or first two points of the
    table.</li>
<li>If the table has only <strong>one row</strong>, no interpolation is performed and
    the function value is just returned independently of the actual time instant.</li>
<li>Via parameters <strong>shiftTime</strong> and <strong>offset</strong> the curve defined
    by the table can be shifted both in time and in the ordinate value.
    The time instants stored in the table are therefore <strong>relative</strong>
    to <strong>shiftTime</strong>.</li>
<li>If time &lt; startTime, no interpolation is performed and the offset
    is used as ordinate value for the output.</li>
<li>If the table has more than one row, the first point in time <strong>always</strong> has to be set to <strong>0</strong>, e.g.,
    <strong>table=[1,1;2,2]</strong> is <strong>illegal</strong>. If you want to
    shift the time table in time use the <strong>shiftTime</strong> parameter instead.</li>
<li>The table is implemented in a numerically sound way by
    generating <strong>time events</strong> at interval boundaries.
    This generates continuously differentiable values for the integrator.</li>
<li>Via parameter <strong>timeScale</strong> the first column of the table array can
    be scaled, e.g., if the table array is given in hours (instead of seconds)
    <strong>timeScale</strong> shall be set to 3600.</li>
</ul>
<p>
Example:
</p>
<pre>
   table = [0, 0;
            1, 0;
            1, 1;
            2, 4;
            3, 9;
            4, 16];
If, e.g., time = 1.0, the output y =  0.0 (before event), 1.0 (after event)
    e.g., time = 1.5, the output y =  2.5,
    e.g., time = 2.0, the output y =  4.0,
    e.g., time = 5.0, the output y = 23.0 (i.e., extrapolation).
</pre>

<p>
<img src=\"modelica://Modelica/Resources/Images/Blocks/Sources/TimeTable.png\"
     alt=\"TimeTable.png\">
</p>

</html>", revisions = "<html>
<h4>Release Notes</h4>
<ul>
<li><em>Oct. 21, 2002</em>
       by Christian Schweiger:<br>
       Corrected interface from
<pre>
    parameter Real table[:, :]=[0, 0; 1, 1; 2, 4];
</pre>
       to
<pre>
    parameter Real table[:, <strong>2</strong>]=[0, 0; 1, 1; 2, 4];
</pre>
       </li>
<li><em>Nov. 7, 1999</em>
       by <a href=\"http://www.robotic.dlr.de/Martin.Otter/\">Martin Otter</a>:<br>
       Realized.</li>
</ul>
</html>"));
    end TimeTable;

    block CombiTimeTable
      "Table look-up with respect to time and linear/periodic extrapolation methods (data from matrix/file)"
      import Modelica.Blocks.Tables.Internal;
      extends Modelica.Blocks.Interfaces.MO(final nout = max([size(columns, 1); size(offset, 1)]));
      parameter Boolean tableOnFile = false
        "= true, if table is defined on file or in function usertab" 
        annotation (Dialog(group = "Table data definition"));
      input Real table[:,:] = fill(0.0, 0, 2)
        "Table matrix (time = first column; e.g., table=[0, 0; 1, 1; 2, 4])" 
        annotation (Dialog(group = "Table data definition", enable = not tableOnFile));
      parameter String tableName = "NoName"
        "Table name on file or in function usertab (see docu)" 
        annotation (Dialog(group = "Table data definition", enable = tableOnFile));
      parameter String fileName = "NoName" "File where matrix is stored" 
        annotation (Dialog(
          group = "Table data definition",
          enable = tableOnFile,
          loadSelector(filter = "Text files (*.txt);;MATLAB MAT-files (*.mat)",
            caption = "Open file in which table is present")));
      parameter Boolean verboseRead = true
        "= true, if info message that file is loading is to be printed" 
        annotation (Dialog(group = "Table data definition", enable = tableOnFile));
      parameter Integer columns[:] = 2:size(table, 2)
        "Columns of table to be interpolated" 
        annotation (Dialog(group = "Table data interpretation",
          groupImage = "modelica://Modelica/Resources/Images/Blocks/Sources/CombiTimeTable.png"));
      parameter Modelica.Blocks.Types.Smoothness smoothness = Modelica.Blocks.Types.Smoothness.LinearSegments
        "Smoothness of table interpolation" 
        annotation (Dialog(group = "Table data interpretation"));
      parameter Modelica.Blocks.Types.Extrapolation extrapolation = Modelica.Blocks.Types.Extrapolation.LastTwoPoints
        "Extrapolation of data outside the definition range" 
        annotation (Dialog(group = "Table data interpretation"));
      parameter Modelica.SIunits.Time timeScale(
        min = Modelica.Constants.eps) = 1 "Time scale of first table column" 
        annotation (Dialog(group = "Table data interpretation"), Evaluate = true);
      parameter Real offset[:] = {0} "Offsets of output signals" 
        annotation (Dialog(group = "Table data interpretation"));
      parameter Modelica.SIunits.Time startTime = 0
        "Output = offset for time < startTime" 
        annotation (Dialog(group = "Table data interpretation"));
      parameter Modelica.SIunits.Time shiftTime = startTime
        "Shift time of first table column" 
        annotation (Dialog(group = "Table data interpretation"));
      parameter Modelica.Blocks.Types.TimeEvents timeEvents = Modelica.Blocks.Types.TimeEvents.Always
        "Time event handling of table interpolation" 
        annotation (Dialog(group = "Table data interpretation", enable = smoothness == Modelica.Blocks.Types.Smoothness.LinearSegments));
      parameter Boolean verboseExtrapolation = false
        "= true, if warning messages are to be printed if time is outside the table definition range" 
        annotation (Dialog(group = "Table data interpretation", enable = extrapolation == Modelica.Blocks.Types.Extrapolation.LastTwoPoints or extrapolation == Modelica.Blocks.Types.Extrapolation.HoldLastPoint));
      final parameter Modelica.SIunits.Time t_min = t_minScaled * timeScale
        "Minimum abscissa value defined in table";
      final parameter Modelica.SIunits.Time t_max = t_maxScaled * timeScale
        "Maximum abscissa value defined in table";
      final parameter Real t_minScaled = Internal.getTimeTableTmin(tableID)
        "Minimum (scaled) abscissa value defined in table";
      final parameter Real t_maxScaled = Internal.getTimeTableTmax(tableID)
        "Maximum (scaled) abscissa value defined in table";
    protected
      final parameter Real p_offset[nout] = (if size(offset, 1) == 1 then ones(nout) * offset[1] else offset)
        "Offsets of output signals";
      parameter Modelica.Blocks.Types.ExternalCombiTimeTable tableID =
        Modelica.Blocks.Types.ExternalCombiTimeTable(
        if tableOnFile then tableName else "NoName",
        if tableOnFile and fileName <> "NoName" and not Modelica.Utilities.Strings.isEmpty(fileName) then fileName else "NoName",
        table,
        startTime / timeScale,
        columns,
        smoothness,
        extrapolation,
        shiftTime / timeScale,
        if smoothness == Modelica.Blocks.Types.Smoothness.LinearSegments then timeEvents else if smoothness == Modelica.Blocks.Types.Smoothness.ConstantSegments then Modelica.Blocks.Types.TimeEvents.Always else Modelica.Blocks.Types.TimeEvents.NoTimeEvents,
        if tableOnFile then verboseRead else false) "External table object";
      discrete Modelica.SIunits.Time nextTimeEvent(start = 0, fixed = true)
        "Next time event instant";
      discrete Real nextTimeEventScaled(start = 0, fixed = true)
        "Next scaled time event instant";
      Real timeScaled "Scaled time";
      function readTableData = Modelica.Blocks.Tables.Internal.readTimeTableData
        "Read table data from text or MATLAB MAT-file";
                               // No longer used, but kept for backward compatibility
    equation
      if tableOnFile then
        assert(tableName <> "NoName",
          "tableOnFile = true and no table name given");
      else
        assert(size(table, 1) > 0 and size(table, 2) > 0,
          "tableOnFile = false and parameter table is an empty matrix");
      end if;
      if verboseExtrapolation and (
      extrapolation == Modelica.Blocks.Types.Extrapolation.LastTwoPoints or 
      extrapolation == Modelica.Blocks.Types.Extrapolation.HoldLastPoint) then
        assert(noEvent(time >= t_min), "
Extrapolation warning: Time (=" + String(time) + ") must be greater or equal
than the minimum abscissa value t_min (=" + String(t_min) + ") defined in the table.
", level = AssertionLevel.warning);
        assert(noEvent(time <= t_max), "
Extrapolation warning: Time (=" + String(time) + ") must be less or equal
than the maximum abscissa value t_max (=" + String(t_max) + ") defined in the table.
", level = AssertionLevel.warning);
      end if;
      timeScaled = time / timeScale;
      when {time >= pre(nextTimeEvent), initial()} then
        nextTimeEventScaled = Internal.getNextTimeEvent(tableID, timeScaled);
        nextTimeEvent = if nextTimeEventScaled < Modelica.Constants.inf then nextTimeEventScaled * timeScale else Modelica.Constants.inf;
      end when;
      if smoothness == Modelica.Blocks.Types.Smoothness.ConstantSegments then
        for i in 1:nout loop
          y[i] = p_offset[i] + Internal.getTimeTableValueNoDer(tableID, i, timeScaled, nextTimeEventScaled, pre(nextTimeEventScaled));
        end for;
      else
        for i in 1:nout loop
          y[i] = p_offset[i] + Internal.getTimeTableValue(tableID, i, timeScaled, nextTimeEventScaled, pre(nextTimeEventScaled));
        end for;
      end if;
      annotation (
        Documentation(info = "<html>
<p>
This block generates an output signal y[:] by <strong>constant</strong>,
<strong>linear</strong> or <strong>cubic Hermite spline interpolation</strong>
in a table. The time points and function values are stored in a matrix
<strong>table[i,j]</strong>, where the first column table[:,1] contains the
time points and the other columns contain the data to be interpolated.
</p>

<p>
<img src=\"modelica://Modelica/Resources/Images/Blocks/Sources/CombiTimeTable.png\"
     alt=\"CombiTimeTable.png\">
</p>

<p>
Via parameter <strong>columns</strong> it can be defined which columns of the
table are interpolated. If, e.g., columns={2,4}, it is assumed that
2 output signals are present and that the first output is computed
by interpolation of column 2 and the second output is computed
by interpolation of column 4 of the table matrix.
The table interpolation has the following properties:
</p>
<ul>
<li>The interpolation interval is found by a binary search where the interval used in the
    last call is used as start interval.</li>
<li>The time points need to be <strong>strictly increasing</strong> for cubic Hermite
    spline interpolation, otherwise <strong>monotonically increasing</strong>.</li>
<li><strong>Discontinuities</strong> are allowed for (constant or) linear interpolation,
    by providing the same time point twice in the table.</li>
<li>Via parameter <strong>smoothness</strong> it is defined how the data is interpolated:
<pre>
  smoothness = 1: Linear interpolation
             = 2: Akima interpolation: Smooth interpolation by cubic Hermite
                  splines such that der(y) is continuous, also if extrapolated.
             = 3: Constant segments
             = 4: Fritsch-Butland interpolation: Smooth interpolation by cubic
                  Hermite splines such that y preserves the monotonicity and
                  der(y) is continuous, also if extrapolated.
             = 5: Steffen interpolation: Smooth interpolation by cubic Hermite
                  splines such that y preserves the monotonicity and der(y)
                  is continuous, also if extrapolated.
</pre></li>
<li>Values <strong>outside</strong> of the table range, are computed by
    extrapolation according to the setting of parameter <strong>extrapolation</strong>:
<pre>
  extrapolation = 1: Hold the first or last value of the table,
                     if outside of the table scope.
                = 2: Extrapolate by using the derivative at the first/last table
                     points if outside of the table scope.
                     (If smoothness is LinearSegments or ConstantSegments
                     this means to extrapolate linearly through the first/last
                     two table points.).
                = 3: Periodically repeat the table data (periodical function).
                = 4: No extrapolation, i.e. extrapolation triggers an error
</pre></li>
<li>If the table has only <strong>one row</strong>, no interpolation is performed and
    the table values of this row are just returned.</li>
<li>Via parameters <strong>shiftTime</strong> and <strong>offset</strong> the curve defined
    by the table can be shifted both in time and in the ordinate value.
    The time instants stored in the table are therefore <strong>relative</strong>
    to <strong>shiftTime</strong>.</li>
<li>If time &lt; startTime, no interpolation is performed and the offset
    is used as ordinate value for all outputs.</li>
<li>The table is implemented in a numerically sound way by
    generating <strong>time events</strong> at interval boundaries, in case of
    interpolation by linear segments.
    This generates continuously differentiable values for the integrator.
    Via parameter <strong>timeEvents</strong> it is defined how the time events are generated:
<pre>
  timeEvents = 1: Always generate time events at interval boundaries
             = 2: Generate time events at discontinuities (defined by duplicated sample points)
             = 3: No time events at interval boundaries
</pre>
    For interpolation by constant segments time events are always generated at interval boundaries.
    For smooth interpolation by cubic Hermite splines no time events are generated at interval boundaries.</li>
<li>Via parameter <strong>timeScale</strong> the first column of the table array can
    be scaled, e.g., if the table array is given in hours (instead of seconds)
    <strong>timeScale</strong> shall be set to 3600.</li>
<li>For special applications it is sometimes needed to know the minimum
    and maximum time instant defined in the table as a parameter. For this
    reason parameters <strong>t_min</strong>/<strong>t_minScaled</strong> and
    <strong>t_max</strong>/<strong>t_maxScaled</strong> are provided and can be
    accessed from the outside of the table object. Whereas <strong>t_min</strong> and
    <strong>t_max</strong> define the scaled abscissa values (using parameter
    <strong>timeScale</strong>) in SIunits.Time, <strong>t_minScaled</strong> and
    <strong>t_maxScaled</strong> define the unitless original abscissa values of
    the table.</li>
</ul>
<p>
Example:
</p>
<pre>
   table = [0, 0;
            1, 0;
            1, 1;
            2, 4;
            3, 9;
            4, 16];
   extrapolation = 2 (default), timeEvents = 2
If, e.g., time = 1.0, the output y =  0.0 (before event), 1.0 (after event)
    e.g., time = 1.5, the output y =  2.5,
    e.g., time = 2.0, the output y =  4.0,
    e.g., time = 5.0, the output y = 23.0 (i.e., extrapolation via last 2 points).
</pre>
<p>
The table matrix can be defined in the following ways:
</p>
<ol>
<li>Explicitly supplied as <strong>parameter matrix</strong> \"table\",
    and the other parameters have the following values:
<pre>
   tableName is \"NoName\" or has only blanks,
   fileName  is \"NoName\" or has only blanks.
</pre></li>
<li><strong>Read</strong> from a <strong>file</strong> \"fileName\" where the matrix is stored as
    \"tableName\". Both text and MATLAB MAT-file format is possible.
    (The text format is described below).
    The MAT-file format comes in four different versions: v4, v6, v7 and v7.3.
    The library supports at least v4, v6 and v7 whereas v7.3 is optional.
    It is most convenient to generate the MAT-file from FreeMat or MATLAB&reg;
    by command
<pre>
   save tables.mat tab1 tab2 tab3
</pre>
    or Scilab by command
<pre>
   savematfile tables.mat tab1 tab2 tab3
</pre>
    when the three tables tab1, tab2, tab3 should be used from the model.<br>
    Note, a fileName can be defined as URI by using the helper function
    <a href=\"modelica://Modelica.Utilities.Files.loadResource\">loadResource</a>.</li>
<li>Statically stored in function \"usertab\" in file \"usertab.c\".
    The matrix is identified by \"tableName\". Parameter
    fileName = \"NoName\" or has only blanks. Row-wise storage is always to be
    preferred as otherwise the table is reallocated and transposed.</li>
</ol>
<p>
When the constant \"NO_FILE_SYSTEM\" is defined, all file I/O related parts of the
source code are removed by the C-preprocessor, such that no access to files takes place.
</p>
<p>
If tables are read from a text file, the file needs to have the
following structure (\"-----\" is not part of the file content):
</p>
<pre>
-----------------------------------------------------
#1
double tab1(6,2)   # comment line
  0   0
  1   0
  1   1
  2   4
  3   9
  4  16
double tab2(6,2)   # another comment line
  0   0
  2   0
  2   2
  4   8
  6  18
  8  32
-----------------------------------------------------
</pre>
<p>
Note, that the first two characters in the file need to be
\"#1\" (a line comment defining the version number of the file format).
Afterwards, the corresponding matrix has to be declared
with type (= \"double\" or \"float\"), name and actual dimensions.
Finally, in successive rows of the file, the elements of the matrix
have to be given. The elements have to be provided as a sequence of
numbers in row-wise order (therefore a matrix row can span several
lines in the file and need not start at the beginning of a line).
Numbers have to be given according to C syntax (such as 2.3, -2, +2.e4).
Number separators are spaces, tab (\\t), comma (,), or semicolon (;).
Several matrices may be defined one after another. Line comments start
with the hash symbol (#) and can appear everywhere.
Text files should either be ASCII or UTF-8 encoded, where UTF-8 encoded strings are only allowed in line comments and an optional UTF-8 BOM at the start of the text file is ignored.
Other characters, like trailing non comments, are not allowed in the file.
</p>
<p>
MATLAB is a registered trademark of The MathWorks, Inc.
</p>
</html>", revisions = "<html>
<p><strong>Release Notes:</strong></p>
<ul>
<li><em>April 09, 2013</em>
       by Thomas Beutlich:<br>
       Implemented as external object.</li>
<li><em>March 31, 2001</em>
       by <a href=\"http://www.robotic.dlr.de/Martin.Otter/\">Martin Otter</a>:<br>
       Used CombiTableTime as a basis and added the
       arguments <strong>extrapolation, columns, startTime</strong>.
       This allows periodic function definitions.</li>
</ul>
</html>"),
        Icon(
          coordinateSystem(preserveAspectRatio = true,
            extent = {{-100.0, -100.0}, {100.0, 100.0}}),
          graphics = {
          Polygon(lineColor = {192, 192, 192},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Solid,
          points = {{-80.0, 90.0}, {-88.0, 68.0}, {-72.0, 68.0}, {-80.0, 90.0}}),
          Line(points = {{-80.0, 68.0}, {-80.0, -80.0}},
          color = {192, 192, 192}),
          Line(points = {{-90.0, -70.0}, {82.0, -70.0}},
          color = {192, 192, 192}),
          Polygon(lineColor = {192, 192, 192},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Solid,
          points = {{90.0, -70.0}, {68.0, -62.0}, {68.0, -78.0}, {90.0, -70.0}}),
          Rectangle(lineColor = {255, 255, 255},
          fillColor = {255, 215, 136},
          fillPattern = FillPattern.Solid,
          extent = {{-48.0, -50.0}, {2.0, 70.0}}),
          Line(points = {{-48.0, -50.0}, {-48.0, 70.0}, {52.0, 70.0}, {52.0, -50.0}, {-48.0, -50.0}, {-48.0, -20.0}, {52.0, -20.0}, {52.0, 10.0}, {-48.0, 10.0}, {-48.0, 40.0}, {52.0, 40.0}, {52.0, 70.0}, {2.0, 70.0}, {2.0, -51.0}})}),
        Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {
          Polygon(
          points = {{-80, 90}, {-88, 68}, {-72, 68}, {-80, 90}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Line(points = {{-80, 68}, {-80, -80}}, color = {95, 95, 95}),
          Line(points = {{-90, -70}, {82, -70}}, color = {95, 95, 95}),
          Polygon(
          points = {{90, -70}, {68, -62}, {68, -78}, {90, -70}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Rectangle(
          extent = {{-20, 90}, {20, -30}},
          lineColor = {255, 255, 255},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Solid),
          Line(points = {{-20, -30}, {-20, 90}, {80, 90}, {80, -30}, {-20, -30}, {-20, 0}, {
          80, 0}, {80, 30}, {-20, 30}, {-20, 60}, {80, 60}, {80, 90}, {20, 90}, {20, -30}}),
          Text(
          extent = {{-71, -42}, {-32, -54}},
          textString = "offset"),
          Polygon(
          points = {{-31, -30}, {-33, -40}, {-28, -40}, {-31, -30}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Polygon(
          points = {{-31, -70}, {-34, -60}, {-29, -60}, {-31, -70}, {-31, -70}},
          lineColor = {95, 95, 95},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid),
          Line(points = {{-31, -31}, {-31, -70}}, color = {95, 95, 95}),
          Line(points = {{-20, -30}, {-20, -70}}, color = {95, 95, 95}),
          Text(
          extent = {{-42, -74}, {6, -84}},
          textString = "startTime"),
          Line(points = {{-20, -30}, {-80, -30}}, color = {95, 95, 95}),
          Text(
          extent = {{-73, 93}, {-44, 74}},
          textString = "y"),
          Text(
          extent = {{66, -81}, {92, -92}},
          textString = "time"),
          Text(
          extent = {{-19, 83}, {20, 68}},
          textString = "time"),
          Text(
          extent = {{21, 82}, {50, 68}},
          textString = "y[1]"),
          Line(points = {{50, 90}, {50, -30}}),
          Line(points = {{80, 0}, {100, 0}}, color = {0, 0, 255}),
          Text(
          extent = {{34, -30}, {71, -42}},
          textString = "columns",
          lineColor = {0, 0, 255}),
          Text(
          extent = {{51, 82}, {80, 68}},
          textString = "y[2]")}));
    end CombiTimeTable;
  end TimeTable;
end Utilities;
