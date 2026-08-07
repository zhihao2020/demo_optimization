within TypicalScensrio;

package BoundaryConditions "边界模型"


  package FluidBoundary "流体边界"
    model BoundaryMdot "流量边界"
      replaceable package Medium = Media.FluidMedia.IdealAir 
         constrainedby TypicalScensrio.Media.FluidMedia.PartialMedium
          "介质" annotation (choicesAllMatching = true, Protection(access = Access.icon));
      parameter Real C_exergy = 0 "成本" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter SI.Power Exergy = 0 "㶲" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter SI.MassFlowRate m_flow = 1 "流量" 
        annotation (Dialog(enable = not use_mflow_in));
      parameter SI.Temperature T = 298.15 "温度" 
        annotation (Dialog(enable = (energyDefinition == "T" and not use_T_in and not use_h_in)));
      parameter SI.SpecificEnthalpy h = 200e3 "比焓" 
        annotation (Dialog(enable = (energyDefinition == "h" and not use_h_in and not use_T_in)));
      parameter Boolean use_mflow_in = false "流量由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Boolean use_T_in = false "温度由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Boolean use_h_in = false "比焓由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      input SI.MassFraction Xi[Medium.nXi] = Medium.reference_X "质量分数" annotation (Dialog(group = "参数"));

      parameter String energyDefinition = "T"
        "使用温度输入还是比焓输入" 
        annotation (choices(
          choice = "h" "比焓输入",
          choice = "T" "温度输入"), Evaluate = true, HideResult = true);

      Modelica.Blocks.Interfaces.RealInput mflow_in if use_mflow_in
        "外部给定压力" annotation (Placement(transformation(
          origin = {-60, 100},
          extent = {{-20, -20}, {20, 20}},
          rotation = 270)));
      Modelica.Blocks.Interfaces.RealInput T_in if use_T_in and energyDefinition == "T"
        "外部给定温度" annotation (Placement(transformation(origin = {60.0, 100.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}},
          rotation = 270.0)));
      Modelica.Blocks.Interfaces.RealInput h_in if use_h_in and energyDefinition == "h"
        "外部给定比焓" annotation (Placement(transformation(origin = {0.0, 100.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}},
          rotation = 270.0)));
      Interfaces.Fluid.FluidPort_b fluidPort(redeclare package Medium = Medium) annotation (Placement(transformation(origin = {99.0, -1.0},
        extent = {{-15.0, -15.0}, {15.0, 15.0}}),
        iconTransformation(origin = {90.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput mflow_in_internal
        "用于连接外部有条件的连接";
      Modelica.Blocks.Interfaces.RealInput T_in_internal
        "用于连接外部有条件的连接";
      Modelica.Blocks.Interfaces.RealInput h_in_internal
        "用于连接外部有条件的连接";
      import SI = Modelica.SIunits;
    equation
      connect(mflow_in, mflow_in_internal);
      connect(T_in, T_in_internal);
      connect(h_in, h_in_internal);

      if not use_mflow_in then
        mflow_in_internal = m_flow;
      end if;
      if not use_T_in then
        T_in_internal = T;
      end if;
      if not use_h_in then
        h_in_internal = h;
      end if;
      //////////////
      fluidPort.m_flow = -mflow_in_internal;
      fluidPort.h_outflow = if use_h_in or energyDefinition == "h" then 
        h_in_internal else Medium.h_pT(fluidPort.p, T_in_internal);
      fluidPort.Xi_outflow = Xi;
      fluidPort.Xk = Medium.setXk_phX(fluidPort.p, h, fluidPort.Xi_outflow);
      C_exergy = fluidPort.C_exergy;
      Exergy = fluidPort.Exergy;
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {0.0, 0.0},
          lineColor = {66, 132, 197},
          fillColor = {85, 170, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-80.0, 80.0}, {80.0, -80.0}}), Text(origin = {2.0, 5.0},
          extent = {{-52.0, 53.0}, {52.0, -53.0}},
          textString = "G",
          textStyle = {TextStyle.None})}), Protection(access = Access.icon));

    end BoundaryMdot;
    model BoundaryPressure "压力边界"
      replaceable package Medium = Media.FluidMedia.IdealAir 
         constrainedby TypicalScensrio.Media.FluidMedia.PartialMedium "介质" 
          annotation (choicesAllMatching = true, Protection(access = Access.icon));
      parameter Real C_exergy = 0 "成本" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter SI.Power Exergy = 0 "㶲" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter SI.AbsolutePressure p = 1e5 "压力" 
        annotation (Dialog(enable = not use_p_in));
      parameter SI.Temperature T = 298.15 "温度" 
        annotation (Dialog(enable = (energyDefinition == "T" and not use_T_in and not use_h_in)));   //and not use_Th_in
      parameter SI.SpecificEnthalpy h = 200e3 "比焓" 
        annotation (Dialog(enable = (energyDefinition == "h" and not use_h_in and not use_T_in)));  //and not use_Th_in
      parameter Boolean use_p_in = false "压力由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Boolean use_T_in = false "温度由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Boolean use_h_in = false "比焓由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter SI.MassFraction Xi[Medium.nXi] = Medium.reference_X "质量分数" annotation (Dialog(group = "参数"));

      parameter String energyDefinition = "T"
        "使用温度输入还是比焓输入" 
        annotation (choices(
          choice = "h" "比焓输入",
          choice = "T" "温度输入"), Evaluate = true, HideResult = true);

      Modelica.Blocks.Interfaces.RealInput p_in if use_p_in "外部给定压力" 
        annotation (Placement(transformation(
          origin = {-60, 100},
          extent = {{-20, -20}, {20, 20}},
          rotation = 270)));
      Modelica.Blocks.Interfaces.RealInput T_in if use_T_in and energyDefinition == "T"
        "外部给定温度" annotation (Placement(transformation(origin = {60.0, 100.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}},
          rotation = 270.0)));
      Modelica.Blocks.Interfaces.RealInput h_in if use_h_in and energyDefinition == "h"
        "外部给定比焓" annotation (Placement(transformation(origin = {0.0, 100.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}},
          rotation = 270.0)));

      Interfaces.Fluid.FluidPort_b fluidPort(redeclare package Medium = Medium) annotation (Placement(transformation(origin = {99.00000000000001, 8.881784197001252e-16},
        extent = {{-13.0, -14.0}, {13.0, 14.0}}),
        iconTransformation(origin = {90.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput p_in_internal
        "用于连接外部有条件的连接";
      Modelica.Blocks.Interfaces.RealInput T_in_internal
        "用于连接外部有条件的连接";
      Modelica.Blocks.Interfaces.RealInput h_in_internal
        "用于连接外部有条件的连接";
      import SI = Modelica.SIunits;
    equation
      connect(p_in, p_in_internal);
      connect(T_in, T_in_internal);
      connect(h_in, h_in_internal);

      if not use_p_in then
        p_in_internal = p;
      end if;
      if not use_T_in then
        T_in_internal = T;
      end if;
      if not use_h_in then
        h_in_internal = h;
      end if;
      //////////////
      fluidPort.p = p_in_internal;
      fluidPort.h_outflow = if use_h_in or energyDefinition == "h" then 
        h_in_internal else Medium.h_pT(p_in_internal, T_in_internal);
      fluidPort.Xi_outflow = Xi;
      fluidPort.Xk = Medium.setXk_phX(p, h, fluidPort.Xi_outflow);
      C_exergy = fluidPort.C_exergy;
      Exergy = fluidPort.Exergy;
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {0.0, 0.0},
          lineColor = {66, 132, 197},
          fillColor = {85, 170, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-80.0, 80.0}, {80.0, -80.0}}), Text(origin = {2.0, 17.0},
          extent = {{-52.0, 53.0}, {52.0, -53.0}},
          textString = "p",
          textStyle = {TextStyle.None}), Text(origin = {11.0, -104.0},
          lineColor = {0, 0, 0},
          extent = {{-75.0, 18.0}, {75.0, -18.0}},
          textString = "%name",
          textStyle = {TextStyle.None},
          textColor = {0, 0, 0},
          horizontalAlignment = LinePattern.None)}), Protection(access = Access.icon));

    end BoundaryPressure;
    model BoundaryPressureMdot "压力流量边界"
      replaceable package Medium = Media.FluidMedia.Water 
         constrainedby TypicalScensrio.Media.FluidMedia.PartialMedium
          "介质" 
          annotation (choicesAllMatching = true, Protection(access = Access.icon));
      parameter Real C_exergy = 0 "成本" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter Modelica.SIunits.Power Exergy = 0 "㶲" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter Modelica.SIunits.MassFlowRate m_flow = 1 "质量流量" 
        annotation (Evaluate = true,
          Dialog(enable = not use_mflow_in));
      parameter Modelica.SIunits.AbsolutePressure p(displayUnit = "bar") = 1e5 "压力" 
        annotation (Evaluate = true,
          Dialog(enable = not use_p_in));
      parameter Modelica.SIunits.SpecificEnthalpy h = 200e3 "比焓" 
        annotation (Evaluate = true,
          Dialog(enable = not use_h_in));
      parameter Boolean use_mflow_in = false "质量流量由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Boolean use_p_in = false "压力由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Boolean use_h_in = false "比焓由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Modelica.SIunits.MassFraction Xi[Medium.nXi] = Medium.reference_X "质量分数" annotation (Dialog(group = "参数"));

      Modelica.Blocks.Interfaces.RealInput mflow_in if use_mflow_in
        "外部给定质量流量" 
        annotation (
          Placement(transformation(
            origin = {-60, 100},
            extent = {{-20, -20}, {20, 20}},
            rotation = 270)));
      Modelica.Blocks.Interfaces.RealInput p_in if use_p_in "外部给定压力" 
        annotation (Placement(transformation(origin = {0.0, 100.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}},
          rotation = 270.0)));
      Modelica.Blocks.Interfaces.RealInput h_in if use_h_in "外部给定比焓" 
        annotation (Placement(transformation(origin = {60.0, 100.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}},
          rotation = 270.0)));
      Interfaces.Fluid.FluidPort_b fluidPort(redeclare package Medium = Medium) 
        annotation (Placement(transformation(origin = {90.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput mflow_in_internal
        "用于连接外部有条件的连接";
      Modelica.Blocks.Interfaces.RealInput p_in_internal
        "用于连接外部有条件的连接";
      Modelica.Blocks.Interfaces.RealInput h_in_internal
        "用于连接外部有条件的连接";
      Modelica.SIunits.Temp_K T(displayUnit = "degC") "介质温度";
    equation
      connect(mflow_in, mflow_in_internal);
      connect(p_in, p_in_internal);

      if not use_mflow_in then
        mflow_in_internal = m_flow;
      end if;
      if not use_p_in then
        p_in_internal = p;
      end if;
      if not use_h_in then
        h_in_internal = h;
      end if;
      //////////////
      fluidPort.p = p_in_internal;
      fluidPort.m_flow = -mflow_in_internal;
      fluidPort.h_outflow = h_in_internal;
      fluidPort.Xi_outflow = Xi;
      T = Medium.T_ph(fluidPort.p, fluidPort.h_outflow);
      fluidPort.Xk = Medium.setXk_phX(fluidPort.p, h, fluidPort.Xi_outflow);
      C_exergy = fluidPort.C_exergy;
      Exergy = fluidPort.Exergy;
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {10.0, 3.552713678800501e-15},
          lineColor = {0, 0, 255},
          fillColor = {85, 170, 255},
          pattern = LinePattern.None,
          fillPattern = FillPattern.Solid,
          lineThickness = 1.0,
          extent = {{-78.0, 72.0}, {78.0, -72.0}}), Text(origin = {12.000000000000007, 2.6645352591003757e-15},
          extent = {{-54.0, 42.5}, {54.0, -42.5}},
          textString = "P_Mdot",
          textStyle = {TextStyle.None})}), Protection(access = Access.icon));

    end BoundaryPressureMdot;
    model BoundaryTank "边界"
      replaceable package Medium = Media.FluidMedia.Water 
         constrainedby TypicalScensrio.Media.FluidMedia.PartialMedium
          "介质" annotation (choicesAllMatching = true, Protection(access = Access.icon));

      parameter Real C_exergy = 0 "成本" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter Modelica.SIunits.Power Exergy = 0 "㶲" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter Modelica.SIunits.SpecificEnthalpy h = 200e3 "比焓" 
        annotation (Evaluate = true,
          Dialog(enable = not use_h_in));
      parameter Boolean use_h_in = false "比焓由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Modelica.SIunits.MassFraction Xi[Medium.nXi] = Medium.reference_X "质量分数" annotation (Dialog(group = "参数"));

      Modelica.Blocks.Interfaces.RealInput h_in if use_h_in "外部给定比焓" 
        annotation (Placement(transformation(origin = {60.0, 100.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}},
          rotation = 270.0)));
      Interfaces.Fluid.FluidPort_b fluidPort(redeclare package Medium = Medium) 
        annotation (Placement(transformation(origin = {78.0, -1.6653345369377348e-16},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput h_in_internal
        "用于连接外部有条件的连接";
    equation
      connect(h_in, h_in_internal);
      if not use_h_in then
        h_in_internal = h;
      end if;
      fluidPort.h_outflow = h_in_internal;
      fluidPort.Xi_outflow = Xi;
      fluidPort.Xk = Medium.setXk_phX(fluidPort.p, h, fluidPort.Xi_outflow);
      C_exergy = fluidPort.C_exergy;
      Exergy = fluidPort.Exergy;

      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {0.0, 0.0},
          lineColor = {66, 132, 197},
          fillColor = {85, 170, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-80.0, 80.0}, {80.0, -80.0}}), Text(origin = {0.0, -1.0},
          extent = {{-52.0, 53.0}, {52.0, -53.0}},
          textString = "T",
          textStyle = {TextStyle.None}), Text(origin = {11.0, -104.0},
          lineColor = {0, 0, 0},
          extent = {{-75.0, 18.0}, {75.0, -18.0}},
          textString = "%name",
          textStyle = {TextStyle.None},
          textColor = {0, 0, 0},
          horizontalAlignment = LinePattern.None)}), Protection(access = Access.icon));

    end BoundaryTank;
    import SI = Modelica.SIunits;
    annotation (Protection(access = Access.icon));
  end FluidBoundary;
  package MechanicalBoundaries "机械边界"
    annotation (Protection(access = Access.icon));
    model BoundarySpeed_N "转速边界"
      parameter SI.AngularVelocity N(displayUnit = "rpm") = 1 "固定转速" 
        annotation (Dialog(enable = not use_N_in));
      parameter Boolean use_N_in = false "转速由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Real C_exergy = 0 "成本" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter SI.Power Exergy = 0 "㶲" 
        annotation (Dialog(tab = "高级", group = "参数"));
      Modelica.Blocks.Interfaces.RealInput N_in(unit = "rpm") if use_N_in "转速 rpm" 
        annotation (Placement(transformation(extent = {{-140, -20}, {-100, 20}}, rotation = 0)));
      TypicalScensrio.Interfaces.Power.PowerPort_a flange 
        annotation (Placement(transformation(origin = {108.99954545454544, -5.9999939393939385},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput N_in_internal
        "用于连接外部有条件的连接";
    equation
      N_in_internal = flange.N;

      connect(N_in, N_in_internal);
      if not use_N_in then
        N_in_internal = N;
      end if;
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Text(origin = {0.0, 90.0},
          lineColor = {0, 0, 255},
          extent = {{-150.0, 20.0}, {150.0, -20.0}},
          textString = "%name",
          textColor = {0, 0, 255}), Line(origin = {-1.0, 31.0},
          points = {{-87.0, -31.0}, {-63.0, -1.0}, {-35.0, 21.0}, {-1.0, 31.0}, {29.0, 25.0}, {49.0, 13.0}, {65.0, -3.0}, {77.0, -17.0}, {87.0, -31.0}},
          thickness = 0.5,
          smooth = Smooth.Bezier), Polygon(origin = {61.5, 29.0},
          fillPattern = FillPattern.Solid,
          points = {{24.5, -29.0}, {4.5, 29.0}, {-24.5, -2.0}, {24.5, -29.0}}), Line(origin = {0.0, -30.0},
          points = {{-30.0, 0.0}, {30.0, 0.0}}), Line(origin = {-20.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {0.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {20.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {5.0, -36.0},
          points = {{-59.0, -6.0}, {-43.0, 8.0}, {-21.0, 20.0}, {-1.0, 22.0}, {17.0, 18.0}, {31.0, 10.0}, {43.0, 0.0}, {51.0, -10.0}, {59.0, -22.0}},
          smooth = Smooth.Bezier), Polygon(origin = {-52.5, -51.0},
          fillPattern = FillPattern.Solid,
          points = {{-8.5, -15.0}, {8.5, 9.0}, {-5.5, 15.0}, {-8.5, -15.0}}), Text(origin = {103.455, 12.9394},
          extent = {{-125.45454545454545, -28.939393939393938}, {-79.45454545454545, 11.060606060606062}},
          textString = "N",
          textStyle = {TextStyle.None}), Line(origin = {0.0, -55.5},
          points = {{0.0, 35.5}, {0.0, -35.5}})}), Documentation(info = "<HTML>
<p>
Model of <b>fixed</b> angular velocity of flange, not dependent on torque.
</p>
</HTML>"),
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0})),
        Documentation(info = "<HTML>
<p>
Partial model of torque that accelerates the flange.
</p>

<p>
If <i>useSupport=true</i>, the support connector is conditionally enabled
and needs to be connected.<br>
If <i>useSupport=false</i>, the support connector is conditionally disabled
and instead the component is internally fixed to ground.
</p>
</html>"), Protection(access = Access.icon));
      C_exergy = flange.C_exergy;
      Exergy = flange.Exergy;
    end BoundarySpeed_N;
    model BoundaryPower "功率边界"
      parameter Modelica.SIunits.Power P = 1 "固定功率" 
        annotation (Dialog(enable = not use_P_in));
      parameter Boolean use_P_in = false "转速由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Real C_exergy = 0 "成本" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter SI.Power Exergy = 0 "㶲" 
        annotation (Dialog(tab = "高级", group = "参数"));
      Modelica.Blocks.Interfaces.RealInput P_in(unit = "W") if use_P_in "功率" 
        annotation (Placement(transformation(extent = {{-140, -20}, {-100, 20}}, rotation = 0)));
      TypicalScensrio.Interfaces.Power.PowerPort_a flange 
        annotation (Placement(transformation(origin = {96.0, -6.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput P_in_internal
        "用于连接外部有条件的连接";
    equation
      flange.P = -P_in_internal;

      connect(P_in, P_in_internal);
      if not use_P_in then
        P_in_internal = P;
      end if;
      C_exergy = flange.C_exergy;
      Exergy = flange.Exergy;
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Text(origin = {103.45454545454545, 12.939393939393938},
          extent = {{-125.45454545454545, -28.939393939393938}, {-79.45454545454545, 11.060606060606062}},
          textString = "P",
          textStyle = {TextStyle.None}), Text(origin = {0.0, 90.0},
          lineColor = {0, 0, 255},
          extent = {{-150.0, 20.0}, {150.0, -20.0}},
          textString = "%name",
          textColor = {0, 0, 255}), Line(origin = {-1.0, 31.0},
          points = {{-87.0, -31.0}, {-63.0, -1.0}, {-35.0, 21.0}, {-1.0, 31.0}, {29.0, 25.0}, {49.0, 13.0}, {65.0, -3.0}, {77.0, -17.0}, {87.0, -31.0}},
          thickness = 0.5,
          smooth = Smooth.Bezier), Polygon(origin = {61.5, 29.0},
          fillPattern = FillPattern.Solid,
          points = {{24.5, -29.0}, {4.5, 29.0}, {-24.5, -2.0}, {24.5, -29.0}}), Line(origin = {0.0, -30.0},
          points = {{-30.0, 0.0}, {30.0, 0.0}}), Line(origin = {0.0, -55.5},
          points = {{0.0, 35.5}, {0.0, -35.5}}), Line(origin = {-20.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {0.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {20.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {5.0, -36.0},
          points = {{-59.0, -6.0}, {-43.0, 8.0}, {-21.0, 20.0}, {-1.0, 22.0}, {17.0, 18.0}, {31.0, 10.0}, {43.0, 0.0}, {51.0, -10.0}, {59.0, -22.0}},
          smooth = Smooth.Bezier), Polygon(origin = {-52.5, -51.0},
          fillPattern = FillPattern.Solid,
          points = {{-8.5, -15.0}, {8.5, 9.0}, {-5.5, 15.0}, {-8.5, -15.0}})}),
        Documentation(info = "<HTML>
<p>
Model of <b>fixed</b> angular velocity of flange, not dependent on torque.
</p>
</HTML>"),
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0})),
        Documentation(info = "<HTML>
<p>
Partial model of torque that accelerates the flange.
</p>

<p>
If <i>useSupport=true</i>, the support connector is conditionally enabled
and needs to be connected.<br>
If <i>useSupport=false</i>, the support connector is conditionally disabled
and instead the component is internally fixed to ground.
</p>
</html>"), Protection(access = Access.icon));

    end BoundaryPower;
    model BoundarySpeed_Power "转速、功率边界"
      parameter SI.AngularVelocity N(displayUnit = "rpm") = 1 "固定转速" 
        annotation (Dialog(enable = not use_N_in));
      parameter Boolean use_N_in = false "转速由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Modelica.SIunits.Power P = 1 "固定功率" 
        annotation (Dialog(enable = not use_P_in));
      parameter Boolean use_P_in = false "转速由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      parameter Real C_exergy = 0 "成本" 
        annotation (Dialog(tab = "高级", group = "参数"));
      parameter SI.Power Exergy = 0 "㶲" 
        annotation (Dialog(tab = "高级", group = "参数"));
      Modelica.Blocks.Interfaces.RealInput N_in(unit = "rpm") if use_N_in "转速" 
        annotation (Placement(transformation(origin = {-120.0, 40.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      Modelica.Blocks.Interfaces.RealInput P_in(unit = "rpm") if use_P_in "功率" 
        annotation (Placement(transformation(origin = {-120.0, -40.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
      TypicalScensrio.Interfaces.Power.PowerPort_a flange 
        annotation (Placement(transformation(origin = {100.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput N_in_internal
        "用于连接外部有条件的连接";
      Modelica.Blocks.Interfaces.RealInput P_in_internal
        "用于连接外部有条件的连接";
    equation
      N_in_internal = flange.N;
      flange.P = -P_in_internal;

      connect(N_in, N_in_internal);
      if not use_N_in then
        N_in_internal = N;
      end if;
      connect(P_in, P_in_internal);
      if not use_P_in then
        P_in_internal = P;
      end if;
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Text(origin = {0.0, 90.0},
          lineColor = {0, 0, 255},
          extent = {{-150.0, 20.0}, {150.0, -20.0}},
          textString = "%name",
          textColor = {0, 0, 255}), Line(origin = {-1.0, 31.0},
          points = {{-87.0, -31.0}, {-63.0, -1.0}, {-35.0, 21.0}, {-1.0, 31.0}, {29.0, 25.0}, {49.0, 13.0}, {65.0, -3.0}, {77.0, -17.0}, {87.0, -31.0}},
          thickness = 0.5,
          smooth = Smooth.Bezier), Polygon(origin = {61.5, 29.0},
          fillPattern = FillPattern.Solid,
          points = {{24.5, -29.0}, {4.5, 29.0}, {-24.5, -2.0}, {24.5, -29.0}}), Line(origin = {0.0, -30.0},
          points = {{-30.0, 0.0}, {30.0, 0.0}}), Line(origin = {-20.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {0.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {20.0, -40.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}}), Line(origin = {5.0, -36.0},
          points = {{-59.0, -6.0}, {-43.0, 8.0}, {-21.0, 20.0}, {-1.0, 22.0}, {17.0, 18.0}, {31.0, 10.0}, {43.0, 0.0}, {51.0, -10.0}, {59.0, -22.0}},
          smooth = Smooth.Bezier), Polygon(origin = {-52.5, -51.0},
          fillPattern = FillPattern.Solid,
          points = {{-8.5, -15.0}, {8.5, 9.0}, {-5.5, 15.0}, {-8.5, -15.0}}), Text(origin = {79.455, 10.9394},
          extent = {{-125.45454545454545, -28.939393939393938}, {-79.45454545454545, 11.060606060606062}},
          textString = "N",
          textStyle = {TextStyle.None}), Line(origin = {0.0, -55.5},
          points = {{0.0, 35.5}, {0.0, -35.5}}), Text(origin = {123.45499999999998, 10.9394},
          extent = {{-125.45454545454545, -28.939393939393938}, {-79.45454545454545, 11.060606060606062}},
          textString = "P",
          textStyle = {TextStyle.None})}),
        Documentation(info = "<HTML>
<p>
Model of <b>fixed</b> angular velocity of flange, not dependent on torque.
</p>
</HTML>"),
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0})),
        Documentation(info = "<HTML>
<p>
Partial model of torque that accelerates the flange.
</p>

<p>
If <i>useSupport=true</i>, the support connector is conditionally enabled
and needs to be connected.<br>
If <i>useSupport=false</i>, the support connector is conditionally disabled
and instead the component is internally fixed to ground.
</p>
</html>"), Protection(access = Access.icon));
      C_exergy = flange.C_exergy;
      Exergy = flange.Exergy;
    end BoundarySpeed_Power;
    import SI = Modelica.SIunits;
  end MechanicalBoundaries;
  package HeatBoundary "热边界"



    annotation (Protection(access = Access.icon));
    model BoundaryAdiabatic "绝热边界"
      parameter Integer nNodes = 1;
      TypicalScensrio.Interfaces.Thermal.HeatPort_a heatPort 
        annotation (Placement(transformation(origin = {100.0, 0.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    equation
      for i in 1:nNodes loop
        heatPort.Q_flow = 0;
      end for;
      annotation (defaultComponentName = "adiabatic",
        Icon(coordinateSystem(preserveAspectRatio = false, extent = {{-100, -100}, {100,
          100}}), graphics = {
          Text(
          extent = {{-150, 150}, {150, 110}},
          textString = "%name",
          lineColor = {0, 0, 255}),
          Rectangle(
          extent = {{100, 100}, {60, -100}},
          lineColor = {0, 0, 0},
          fillColor = {175, 175, 175},
          fillPattern = FillPattern.Solid),
          Line(
          points = {{0, -16}, {60, 20}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, 24}, {60, 60}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, 64}, {60, 100}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, -56}, {60, -20}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, -96}, {60, -60}},
          color = {255, 0, 0},
          thickness = 0.5)}),
        Documentation(info = "<html>
<p>This model defines an adiabatic boundary condition (Q_flow = 0) for all nodes.</p>
</html>"),
        Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {
          Rectangle(
          extent = {{100, 100}, {60, -100}},
          lineColor = {0, 0, 0},
          fillColor = {175, 175, 175},
          fillPattern = FillPattern.Solid),
          Line(
          points = {{0, -16}, {60, 20}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, 24}, {60, 60}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, 64}, {60, 100}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, -56}, {60, -20}},
          color = {255, 0, 0},
          thickness = 0.5),
          Line(
          points = {{0, -96}, {60, -60}},
          color = {255, 0, 0},
          thickness = 0.5)}), Protection(access = Access.icon));

    end BoundaryAdiabatic;
    model BoundaryTemperature "温度边界"
      parameter Integer n = 1 "分段数";
      parameter SI.Temperature T = 293.15 "温度" 
        annotation (Dialog(enable = not use_T_in));
      parameter Boolean use_T_in = false "热流量由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      Modelica.Blocks.Interfaces.RealInput T_in(unit = "W") if use_T_in 
        annotation (Placement(transformation(origin = {-120.0, 0.0},
          extent = {{20.0, -20.0}, {-20.0, 20.0}},
          rotation = 180.0)));

      Interfaces.Thermal.HeatPort_b port[n] annotation (Placement(transformation(extent = {{90,
        -10}, {110, 10}}, rotation = 0)));
    protected
      Modelica.Blocks.Interfaces.RealInput T_in_internal
        "用于连接外部有条件的连接";
    equation
      connect(T_in, T_in_internal);
      if not use_T_in then
        T_in_internal = T;
      end if;
      port.T = fill(T_in_internal, n);
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Rectangle(origin = {80.0, 0.0},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid,
          extent = {{-10.0, 40.0}, {10.0, -40.0}}), Text(origin = {0.0, 130.0},
          lineColor = {0, 0, 255},
          extent = {{-150.0, 20.0}, {150.0, -20.0}},
          textString = "%name",
          textColor = {0, 0, 255}), Text(origin = {0.0, -125.0},
          extent = {{-150.0, 15.0}, {150.0, -15.0}},
          textString = "T=%T"), Rectangle(origin = {0.0, 0.0},
          fillColor = {159, 159, 223},
          pattern = LinePattern.None,
          fillPattern = FillPattern.Backward,
          extent = {{-100.0, 100.0}, {100.0, -100.0}}), Text(origin = {-50.0, -50.0},
          extent = {{50.0, 50.0}, {-50.0, -50.0}},
          textString = "K"), Line(origin = {2.0, 0.0},
          points = {{-54.0, 0.0}, {54.0, 0.0}},
          color = {191, 0, 0},
          thickness = 0.5), Polygon(origin = {70.0, 0.0},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid,
          points = {{-20.0, -20.0}, {-20.0, 20.0}, {20.0, 0.0}, {-20.0, -20.0}})}),
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, -0.5},
          fillColor = {159, 159, 223},
          pattern = LinePattern.None,
          fillPattern = FillPattern.Backward,
          extent = {{-100.0, 100.5}, {100.0, -100.5}}), Line(origin = {2.0, 0.0},
          points = {{-54.0, 0.0}, {54.0, 0.0}},
          color = {191, 0, 0},
          thickness = 0.5), Text(origin = {-50.0, -50.0},
          extent = {{50.0, 50.0}, {-50.0, -50.0}},
          textString = "K"), Polygon(origin = {71.0, 0.0},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid,
          points = {{-19.0, -20.0}, {-19.0, 20.0}, {19.0, 0.0}, {-19.0, -20.0}})}));
    end BoundaryTemperature;
    model BoundaryHeatFlow "热流边界"
      parameter Integer n = 1 "分段数";
      parameter SI.HeatFlowRate Q_flow = 100 "热流量" 
        annotation (Dialog(enable = not use_Qflow_in));
      parameter Boolean use_Qflow_in = false "热流量由外部接口输入" 
        annotation (Dialog(group = "数据来源选项"), Evaluate = true, HideResult = true, choices(checkBox = true));
      TypicalScensrio.Interfaces.Thermal.HeatPort_a heatPort 
        annotation (Placement(transformation(origin = {100.0, 1.9999999999999964},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealInput Q_flow_in(unit = "W") if use_Qflow_in 
        annotation (Placement(transformation(origin = {-90.0, 0.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}})));
    protected
      Modelica.Blocks.Interfaces.RealInput Qflow_in_internal
        "用于连接外部有条件的连接";
    equation
      connect(Q_flow_in, Qflow_in_internal);
      if not use_Qflow_in then
        Qflow_in_internal = Q_flow;
      end if;
      heatPort.Q_flow = -Qflow_in_internal;

      annotation (
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {
          Line(
          points = {{-60, -20}, {40, -20}},
          color = {191, 0, 0},
          thickness = 0.5),
          Line(
          points = {{-60, 20}, {40, 20}},
          color = {191, 0, 0},
          thickness = 0.5),
          Line(
          points = {{-80, 0}, {-60, -20}},
          color = {191, 0, 0},
          thickness = 0.5),
          Line(
          points = {{-80, 0}, {-60, 20}},
          color = {191, 0, 0},
          thickness = 0.5),
          Polygon(
          points = {{40, 0}, {40, 40}, {70, 20}, {40, 0}},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid),
          Polygon(
          points = {{40, -40}, {40, 0}, {70, -20}, {40, -40}},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid),
          Rectangle(
          extent = {{70, 40}, {90, -40}},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid),
          Text(
          extent = {{-150, 100}, {150, 60}},
          textString = "%name",
          lineColor = {0, 0, 255})}),
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Line(origin = {4.0, -20.0},
          points = {{-64.0, 0.0}, {64.0, 0.0}},
          color = {191, 0, 0},
          thickness = 0.5), Line(origin = {4.0, 20.0},
          points = {{-64.0, 0.0}, {64.0, 0.0}},
          color = {191, 0, 0},
          thickness = 0.5), Line(origin = {-70.0, -10.0},
          points = {{-10.0, 10.0}, {10.0, -10.0}},
          color = {191, 0, 0},
          thickness = 0.5), Line(origin = {-70.0, 10.0},
          points = {{-10.0, -10.0}, {10.0, 10.0}},
          color = {191, 0, 0},
          thickness = 0.5), Polygon(origin = {75.0, 20.0},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid,
          points = {{-15.0, -20.0}, {-15.0, 20.0}, {15.0, 0.0}, {-15.0, -20.0}}), Polygon(origin = {75.0, -20.0},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid,
          points = {{-15.0, -20.0}, {-15.0, 20.0}, {15.0, 0.0}, {-15.0, -20.0}})}), Protection(access = Access.icon));

    end BoundaryHeatFlow;
    import SI = Modelica.SIunits;
    model Convection "对流边界-圆柱"
      //-接口a与b之间采用圆柱坐标的有限差分法
      // Convection boundary condition for finite difference between port_a and port_b for Cylindrical Coordinates
      parameter Integer nNodes(min = 2) = 2 "节点数";
      parameter SI.CoefficientOfHeatTransfer alphas[nNodes] = fill(0, nNodes)
        "对流换热系数";
      parameter Boolean isAxial = true "true-轴向对流；false-径向对流" 
        annotation (Evaluate = true);

      parameter Boolean isVolCentered = false
        "true-使用轴向控制体中心进行求解" 
        annotation (Dialog(enable = isAxial), Evaluate = true);

      parameter Boolean isInner = false
        "true-内径计算面积；false-外径计算面积" 
        annotation (Dialog(enable = isAxial), Evaluate = true);

      input SI.Length r_inner = 1 "内径" annotation (Dialog(group = "几何参数", enable = (if not isAxial then true else isInner)));
      input SI.Length r_outer = 1 "外径" annotation (Dialog(group = "几何参数", enable = (if not isAxial then true else not isInner)));
      input SI.Length length = 1 "轴向长度" annotation (Dialog(group = "几何参数", enable = isAxial));

      SI.Area A;
      SI.Area A_node[nNodes];
      SI.Length dxr;
      SI.Length[nNodes] xr;

      Interfaces.Thermal.HeatPort_a port_a[nNodes] annotation (Placement(
        transformation(
          extent = {{-10, -10}, {10, 10}},
          rotation = -90,
          origin = {-100, 0}), iconTransformation(
          extent = {{-40, -10}, {40, 10}},
          rotation = -90,
          origin = {-110, 0})));
      Interfaces.Thermal.HeatPort_b port_b[nNodes] annotation (Placement(
        transformation(
          extent = {{-10, -10}, {10, 10}},
          rotation = -90,
          origin = {100, 0}), iconTransformation(
          extent = {{-40, -10}, {40, 10}},
          rotation = -90,
          origin = {110, 0})));
    equation
      if isAxial then
        // 轴向对流时，计算节点的面积
        if isInner then
          A = 2 * Modelica.Constants.pi * length * r_inner;
        else
          A = 2 * Modelica.Constants.pi * length * r_outer;
        end if;
        if isVolCentered then
          dxr = length / (nNodes);
          xr[1:nNodes] = {dxr * (i - 1) + dxr / 2 for i in 1:nNodes};
          A_node[1:nNodes] = A / nNodes * ones(nNodes);
        else
          dxr = length / (nNodes - 1);
          xr[1:nNodes] = {dxr * (i - 1) for i in 1:nNodes};
          A_node[1] = 0.5 * A / (nNodes - 1);
          A_node[2:nNodes - 1] = A / (nNodes - 1) * ones(nNodes - 2);
          A_node[nNodes] = A_node[1];
        end if;
      else
        //径向对流时，计算节点的面积
        A = Modelica.Constants.pi * (r_outer ^ 2 - r_inner ^ 2);
        dxr = (r_outer - r_inner) / (nNodes - 1);

        xr[1:nNodes] = {dxr * (i - 1) + r_inner for i in 1:nNodes};
        A_node[1] = Modelica.Constants.pi * xr[1] * dxr;
        A_node[2:nNodes - 1] = {2 * Modelica.Constants.pi * xr[i] * dxr for i in 2:nNodes - 1};
        A_node[nNodes] = Modelica.Constants.pi * xr[end] * dxr;
      end if;
      for i in 1:nNodes loop
        port_a[i].Q_flow + port_b[i].Q_flow = 0 "能量平衡";
        port_a[i].Q_flow = alphas[i] * A_node[i] * (port_a[i].T - port_b[i].T)
          "换热量计算";
      end for;
      annotation (Diagram(coordinateSystem(preserveAspectRatio = false, extent = {{-100,
        -100}, {100, 100}}), graphics = {
        Rectangle(
        extent = {{-66, 84}, {94, -76}},
        lineColor = {255, 255, 255},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid),
        Rectangle(
        extent = {{-94, 84}, {-64, -76}},
        lineColor = {0, 0, 0},
        fillColor = {192, 192, 192},
        fillPattern = FillPattern.Backward),
        Text(
        extent = {{-154, -86}, {146, -126}},
        textString = "%name",
        lineColor = {0, 0, 255}),
        Line(points = {{96, 4}, {96, 4}}, color = {0, 127, 255}),
        Line(points = {{-64, 24}, {72, 24}}, color = {191, 0, 0}),
        Line(points = {{-64, -16}, {72, -16}}, color = {191, 0, 0}),
        Line(points = {{-38, 84}, {-38, -76}}, color = {0, 127, 255}),
        Line(points = {{2, 84}, {2, -76}}, color = {0, 127, 255}),
        Line(points = {{36, 84}, {36, -76}}, color = {0, 127, 255}),
        Line(points = {{72, 84}, {72, -76}}, color = {0, 127, 255}),
        Line(points = {{-38, -76}, {-48, -56}}, color = {0, 127, 255}),
        Line(points = {{-38, -76}, {-28, -56}}, color = {0, 127, 255}),
        Line(points = {{2, -76}, {-8, -56}}, color = {0, 127, 255}),
        Line(points = {{2, -76}, {12, -56}}, color = {0, 127, 255}),
        Line(points = {{36, -76}, {26, -56}}, color = {0, 127, 255}),
        Line(points = {{36, -76}, {46, -56}}, color = {0, 127, 255}),
        Line(points = {{72, -76}, {62, -56}}, color = {0, 127, 255}),
        Line(points = {{72, -76}, {82, -56}}, color = {0, 127, 255}),
        Line(points = {{52, -26}, {72, -16}}, color = {191, 0, 0}),
        Line(points = {{52, -6}, {72, -16}}, color = {191, 0, 0}),
        Line(points = {{52, 14}, {72, 24}}, color = {191, 0, 0}),
        Line(points = {{52, 34}, {72, 24}}, color = {191, 0, 0})}), Icon(
          coordinateSystem(preserveAspectRatio = false, extent = {{-100, -100}, {100, 100}}),
          graphics = {
          Rectangle(
          extent = {{-64, 84}, {96, -76}},
          lineColor = {255, 255, 255},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid),
          Rectangle(
          extent = {{-92, 84}, {-62, -76}},
          lineColor = {0, 0, 0},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Backward),
          Text(
          extent = {{-152, -86}, {148, -126}},
          textString = "%name",
          lineColor = {0, 0, 255}),
          Line(points = {{98, 4}, {98, 4}}, color = {0, 127, 255}),
          Line(points = {{-62, 24}, {74, 24}}, color = {191, 0, 0}),
          Line(points = {{-62, -16}, {74, -16}}, color = {191, 0, 0}),
          Line(points = {{-36, 84}, {-36, -76}}, color = {0, 127, 255}),
          Line(points = {{4, 84}, {4, -76}}, color = {0, 127, 255}),
          Line(points = {{38, 84}, {38, -76}}, color = {0, 127, 255}),
          Line(points = {{74, 84}, {74, -76}}, color = {0, 127, 255}),
          Line(points = {{-36, -76}, {-46, -56}}, color = {0, 127, 255}),
          Line(points = {{-36, -76}, {-26, -56}}, color = {0, 127, 255}),
          Line(points = {{4, -76}, {-6, -56}}, color = {0, 127, 255}),
          Line(points = {{4, -76}, {14, -56}}, color = {0, 127, 255}),
          Line(points = {{38, -76}, {28, -56}}, color = {0, 127, 255}),
          Line(points = {{38, -76}, {48, -56}}, color = {0, 127, 255}),
          Line(points = {{74, -76}, {64, -56}}, color = {0, 127, 255}),
          Line(points = {{74, -76}, {84, -56}}, color = {0, 127, 255}),
          Line(points = {{54, -26}, {74, -16}}, color = {191, 0, 0}),
          Line(points = {{54, -6}, {74, -16}}, color = {191, 0, 0}),
          Line(points = {{54, 14}, {74, 24}}, color = {191, 0, 0}),
          Line(points = {{54, 34}, {74, 24}}, color = {191, 0, 0})}),
        Documentation);
    end Convection;
    model ConvectiveResistor
      "Lumped thermal element for heat convection (dT = Rc*Q_flow)"
      Modelica.SIunits.HeatFlowRate Q_flow "Heat flow rate from solid -> fluid";
      Modelica.SIunits.TemperatureDifference dT "= solid.T - fluid.T";
      parameter Modelica.SIunits.CoefficientOfHeatTransfer h = 1
        "Signal representing the convective thermal resistance in [K/W]" 
        annotation (Placement(transformation(
          origin = {0, 100},
          extent = {{-20, -20}, {20, 20}},
          rotation = 270)));
      parameter Modelica.SIunits.Volume V = 1 "容积";
      final parameter SI.Area A = pi * (D + 2 * e) * L;
      final parameter Modelica.SIunits.Length L = 4 * V / (pi * D ^ 2) "管长";
      parameter Modelica.SIunits.Diameter D = 0.2 "内径";
      parameter Modelica.SIunits.Thickness e = 2.e-3 "壁厚";
      Interfaces.Thermal.HeatPort_a solid annotation (Placement(transformation(extent = {{
        -110, -10}, {-90, 10}})));
      Interfaces.Thermal.HeatPort_b fluid annotation (Placement(transformation(extent = {{
        90, -10}, {110, 10}})));
    equation
      dT = solid.T - fluid.T;
      solid.Q_flow = Q_flow;
      fluid.Q_flow = -Q_flow;
      Q_flow = h * A * dT;
      annotation (
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {
          Rectangle(
          extent = {{-62, 80}, {98, -80}},
          lineColor = {255, 255, 255},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid),
          Rectangle(
          extent = {{-90, 80}, {-60, -80}},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Forward),
          Text(
          extent = {{-150, -90}, {150, -130}},
          textString = "%name",
          lineColor = {0, 0, 255}),
          Line(points = {{100, 0}, {100, 0}}, color = {0, 127, 255}),
          Line(points = {{-60, 20}, {76, 20}}, color = {191, 0, 0}),
          Line(points = {{-60, -20}, {76, -20}}, color = {191, 0, 0}),
          Line(points = {{-34, 80}, {-34, -80}}, color = {0, 127, 255}),
          Line(points = {{6, 80}, {6, -80}}, color = {0, 127, 255}),
          Line(points = {{40, 80}, {40, -80}}, color = {0, 127, 255}),
          Line(points = {{76, 80}, {76, -80}}, color = {0, 127, 255}),
          Line(points = {{-34, -80}, {-44, -60}}, color = {0, 127, 255}),
          Line(points = {{-34, -80}, {-24, -60}}, color = {0, 127, 255}),
          Line(points = {{6, -80}, {-4, -60}}, color = {0, 127, 255}),
          Line(points = {{6, -80}, {16, -60}}, color = {0, 127, 255}),
          Line(points = {{40, -80}, {30, -60}}, color = {0, 127, 255}),
          Line(points = {{40, -80}, {50, -60}}, color = {0, 127, 255}),
          Line(points = {{76, -80}, {66, -60}}, color = {0, 127, 255}),
          Line(points = {{76, -80}, {86, -60}}, color = {0, 127, 255}),
          Line(points = {{56, -30}, {76, -20}}, color = {191, 0, 0}),
          Line(points = {{56, -10}, {76, -20}}, color = {191, 0, 0}),
          Line(points = {{56, 10}, {76, 20}}, color = {191, 0, 0}),
          Line(points = {{56, 30}, {76, 20}}, color = {191, 0, 0}),
          Text(
          extent = {{22, 124}, {92, 98}},
          textString = "Rc")}),
        Documentation(info = "<html>
<p>
This is a model of linear heat convection, e.g., the heat transfer between a plate and the surrounding air; same as the
<a href=\"modelica://Modelica.Thermal.HeatTransfer.Components.Convection\">Convection</a> component
but using the convective resistance instead of the convective conductance as an input.
This is advantageous for series connections of ConvectiveResistors,
especially if it shall be allowed that a convective resistance is defined to be zero (i.e. no temperature difference).
</p>
</html>"), Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100,
          100}}), graphics = {
          Rectangle(
          extent = {{-90, 80}, {-60, -80}},
          fillColor = {192, 192, 192},
          fillPattern = FillPattern.Forward),
          Line(points = {{100, 0}, {100, 0}}, color = {0, 127, 255}),
          Line(points = {{100, 0}, {100, 0}}, color = {0, 127, 255}),
          Line(points = {{100, 0}, {100, 0}}, color = {0, 127, 255}),
          Text(
          extent = {{-40, 40}, {80, 20}},
          lineColor = {255, 0, 0},
          textString = "Q_flow"),
          Line(points = {{-60, 20}, {76, 20}}, color = {191, 0, 0}),
          Line(points = {{-60, -20}, {76, -20}}, color = {191, 0, 0}),
          Line(points = {{-34, 80}, {-34, -80}}, color = {0, 127, 255}),
          Line(points = {{6, 80}, {6, -80}}, color = {0, 127, 255}),
          Line(points = {{40, 80}, {40, -80}}, color = {0, 127, 255}),
          Line(points = {{76, 80}, {76, -80}}, color = {0, 127, 255}),
          Line(points = {{-34, -80}, {-44, -60}}, color = {0, 127, 255}),
          Line(points = {{-34, -80}, {-24, -60}}, color = {0, 127, 255}),
          Line(points = {{6, -80}, {-4, -60}}, color = {0, 127, 255}),
          Line(points = {{6, -80}, {16, -60}}, color = {0, 127, 255}),
          Line(points = {{40, -80}, {30, -60}}, color = {0, 127, 255}),
          Line(points = {{40, -80}, {50, -60}}, color = {0, 127, 255}),
          Line(points = {{76, -80}, {66, -60}}, color = {0, 127, 255}),
          Line(points = {{76, -80}, {86, -60}}, color = {0, 127, 255}),
          Line(points = {{56, -30}, {76, -20}}, color = {191, 0, 0}),
          Line(points = {{56, -10}, {76, -20}}, color = {191, 0, 0}),
          Line(points = {{56, 10}, {76, 20}}, color = {191, 0, 0}),
          Line(points = {{56, 30}, {76, 20}}, color = {191, 0, 0})}));
    end ConvectiveResistor;
  end HeatBoundary;
  annotation (Protection(access = Access.icon));
end BoundaryConditions;
