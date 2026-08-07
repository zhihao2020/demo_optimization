within TypicalScensrio;

package Interfaces "接口模型"


  package Fluid "流体接口"
    connector FluidPort "基础流体接口"
      replaceable package Medium = TypicalScensrio.Media.FluidMedia.PartialMedium 
         constrainedby Media.FluidMedia.PartialMedium "工质模型" 
          annotation (choicesAllMatching = true, Protection(access = Access.icon));


      flow SI.MassFlowRate m_flow "质量流量";
      SI.AbsolutePressure p "压力";
      stream SI.SpecificEnthalpy h_outflow "比焓，h_outflow[1]表示蒸汽/水比焓，h_outflow[2]表示空气比焓";
      stream Real C_exergy "成本流";
      stream SI.Power Exergy "㶲";
      stream SI.MassFraction Xi_outflow[Medium.nXi] "质量分数，Xi_outflow[1]表示蒸汽/水质量分数，Xi_outflow[2]表示空气质量分数";
      stream SI.MassFraction Xk[Medium.nXi + 1] "质量分数，Xk_outflow[1]表示蒸汽质量分数，Xk_outflow[2]表示液态水质量分数, Xk_outflow[3]表示空气质量分数";

      annotation (Protection(access = Access.icon));
    end FluidPort;


    connector FluidPort_a "流体接口a"
      extends FluidPort;
      annotation (defaultComponentName = "port_A",
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          preserveAspectRatio = false,
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {0.0, 0.0},
          lineColor = {0, 0, 0},
          fillColor = {0, 127, 255},
          fillPattern = FillPattern.Solid,
          lineThickness = 1.0,
          extent = {{-40.0, 40.0}, {40.0, -40.0}}), Text(origin = {0.0, 80.0},
          extent = {{-150.0, 30.0}, {150.0, -30.0}},
          textString = "%name")}),
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          preserveAspectRatio = false,
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {0.0, 0.0},
          lineColor = {0, 127, 255},
          fillColor = {0, 127, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-100.0, 100.0}, {100.0, -100.0}}), Ellipse(origin = {0.0, 0.0},
          lineColor = {0, 0, 0},
          fillColor = {0, 127, 255},
          fillPattern = FillPattern.Solid,
          lineThickness = 1.0,
          extent = {{-100.0, 100.0}, {100.0, -100.0}})}), Protection(access = Access.icon));

    end FluidPort_a;
    annotation (Protection(access = Access.icon));
    connector FluidPort_b "流体接口b"
      extends FluidPort;
      annotation (defaultComponentName = "port_B",
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          preserveAspectRatio = false,
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {0.0, 0.0},
          lineColor = {0, 0, 0},
          fillColor = {0, 127, 255},
          fillPattern = FillPattern.Solid,
          lineThickness = 1.0,
          extent = {{-40.0, 40.0}, {40.0, -40.0}}), Text(origin = {0.0, 80.0},
          extent = {{-150.0, 30.0}, {150.0, -30.0}},
          textString = "%name"), Ellipse(origin = {0.0, 0.0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-35.0, 35.0}, {35.0, -35.0}})}), Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
            preserveAspectRatio = false,
            grid = {2.0, 2.0}), graphics = {Ellipse(origin = {0.0, 0.0},
          lineColor = {0, 127, 255},
          fillColor = {0, 127, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-100.0, 100.0}, {100.0, -100.0}}), Ellipse(origin = {0.0, 0.0},
          lineColor = {0, 0, 0},
          fillColor = {0, 127, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-100.0, 100.0}, {100.0, -100.0}}), Ellipse(origin = {0.0, 0.0},
          lineColor = {0, 0, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-80.0, 80.0}, {80.0, -80.0}})}), Protection(access = Access.icon));

    end FluidPort_b;
    import SI = Modelica.SIunits;
  end Fluid;
  package Thermal "热学接口"
    annotation (Diagram(coordinateSystem(extent = {{-140.0, -100.0}, {140.0, 100.0}},
      preserveAspectRatio = false,
      grid = {2.0, 2.0})),
      Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        preserveAspectRatio = false,
        grid = {2.0, 2.0})), Protection(access = Access.icon));

    partial connector HeatPort "基础热接口"
      SI.Temperature T "接口温度";
      flow SI.HeatFlowRate Q_flow
        "接口热流量（流入为正，流出为负）";
      annotation (Documentation(info = "<html>

</html>"), Protection(access = Access.icon));

    end HeatPort;

    connector HeatPort_a "热流接口a"
      extends HeatPort;
      annotation (defaultComponentName = "heatport_a",
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {Rectangle(
          extent = {{-100, 100}, {100, -100}},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100},
          {100, 100}}), graphics = {Rectangle(
          extent = {{-50, 50}, {50, -50}},
          lineColor = {191, 0, 0},
          fillColor = {191, 0, 0},
          fillPattern = FillPattern.Solid), Text(
          extent = {{-120, 120}, {100, 60}},
          lineColor = {191, 0, 0},
          textString = "%name")}), Protection(access = Access.icon));

    end HeatPort_a;

    connector HeatPort_b "热流接口b"
      extends HeatPort;
      annotation (defaultComponentName = "heatport_b",
        Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100},
          {100, 100}}), graphics = {Rectangle(
          extent = {{-50, 50}, {50, -50}},
          lineColor = {191, 0, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid), Text(
          extent = {{-100, 120}, {120, 60}},
          lineColor = {191, 0, 0},
          textString = "%name")}),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {Rectangle(
          extent = {{-100, 100}, {100, -100}},
          lineColor = {191, 0, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid)}), Protection(access = Access.icon));

    end HeatPort_b;
    import SI = Modelica.SIunits;
  end Thermal;
  package Mechanics "机械接口"

    connector Flange "基础机械接口"

      SI.Angle phi "绝对旋转角度";
      flow SI.Torque tau "扭矩";
      annotation (
        Documentation(info = "<html>
<p>
This is a connector for 1D rotational mechanical systems.
It has no icon definition and is only used by inheritance from
flange connectors to define different icons.
</p>
<p>
The following variables are defined in this connector:
</p>

<blockquote><pre>
phi: Absolute rotation angle of the flange in [rad].
tau: Cut-torque in the flange in [Nm].
</pre></blockquote>
</html>"),
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0})), Protection(access = Access.icon));

    end Flange;
    connector Flange_a "机械接口a"
      extends Flange;

      annotation (
        defaultComponentName = "flange_a",
        Documentation(info = "<html>
<p>
This is a connector for 1-dim. translational mechanical systems which represents
a mechanical flange. In the cut plane of
the flange a unit vector n, called flange axis, is defined which is directed
INTO the cut plane, i. e. from left to right. All vectors in the cut plane are
resolved with respect to
this unit vector. E.g. force f characterizes a vector which is directed in
the direction of n with value equal to f. When this flange is connected to
other 1D translational flanges, this means that the axes vectors of the connected
flanges are identical.
</p>
<p>
The following variables are transported through this connector:
</p>

<blockquote><pre>
s: Absolute position of the flange in [m]. A positive translation
   means that the flange is translated along the flange axis.
f: Cut-force in direction of the flange axis in [N].
</pre></blockquote>
</html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {Rectangle(
          extent = {{-100, -100}, {100, 100}},
          lineColor = {0, 127, 0},
          fillColor = {0, 127, 0},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100},
          {100, 100}}), graphics = {Rectangle(
          extent = {{-40, -40}, {40, 40}},
          lineColor = {0, 127, 0},
          fillColor = {0, 127, 0},
          fillPattern = FillPattern.Solid), Text(
          extent = {{-160, 110}, {40, 50}},
          lineColor = {0, 127, 0},
          textString = "%name")}), Protection(access = Access.icon));

    end Flange_a;
    connector Flange_b "机械接口b"
      extends Flange;

      annotation (
        defaultComponentName = "flange_b",
        Documentation(info = "<html>
<p>
This is a connector for 1-dim. translational mechanical systems which represents
a mechanical flange. In the cut plane of
the flange a unit vector n, called flange axis, is defined which is directed
OUT OF the cut plane. All vectors in the cut plane are resolved with respect to
this unit vector. E.g. force f characterizes a vector which is directed in
the direction of n with value equal to f. When this flange is connected to
other 1D translational flanges, this means that the axes vectors of the connected
flanges are identical.
</p>
<p>
The following variables are transported through this connector:
</p>

<blockquote><pre>
s: Absolute position of the flange in [m]. A positive translation
   means that the flange is translated along the flange axis.
f: Cut-force in direction of the flange axis in [N].
</pre></blockquote>
</html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {
          100, 100}}), graphics = {Rectangle(
          extent = {{-100, -100}, {100, 100}},
          lineColor = {0, 127, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100},
          {100, 100}}), graphics = {Rectangle(
          extent = {{-40, -40}, {40, 40}},
          lineColor = {0, 127, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid), Text(
          extent = {{-40, 110}, {160, 50}},
          lineColor = {0, 127, 0},
          textString = "%name")}), Protection(access = Access.icon));

    end Flange_b;
    annotation (Protection(access = Access.icon));
    import SI = Modelica.SIunits;
  end Mechanics;
  package Electrical "电学接口"
    connector Pin "基础电学接口"
      SI.Voltage v "接口电压" annotation (
        unassignedMessage = "An electrical potential cannot be uniquely calculated.
The reason could be that
- a ground object is missing (Modelica.Electrical.Analog.Basic.Ground)
  to define the zero potential of the electrical circuit, or
- a connector of an electrical component is not connected.");
      flow SI.Current i "接口电流" annotation (
        unassignedMessage = "An electrical current cannot be uniquely calculated.
The reason could be that
- a ground object is missing (Modelica.Electrical.Analog.Basic.Ground)
  to define the zero potential of the electrical circuit, or
- a connector of an electrical component is not connected.");
      annotation (defaultComponentName = "pin",
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100,
          100}}), graphics = {Rectangle(
          extent = {{-100, 100}, {100, -100}},
          lineColor = {0, 0, 255},
          fillColor = {0, 0, 255},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0})),
        Documentation(revisions = "<html>
<ul>
<li><i> 1998   </i>
       by Christoph Clauss<br> initially implemented<br>
       </li>
</ul>
</html>", info = "<html>
<p>Pin is the basic electric connector. It includes the voltage which consists between the pin and the ground node. The ground node is the node of (any) ground device (Modelica.Electrical.Basic.Ground). Furthermore, the pin includes the current, which is considered to be <b>positive</b> if it is flowing at the pin<b> into the device</b>.</p>
</html>"), Protection(access = Access.icon));

    end Pin;
    connector PositivePin "正极接口"
      extends Pin;
      annotation (defaultComponentName = "pin_p",
        Documentation(info = "<html>
<p>Connectors PositivePin and NegativePin are nearly identical. The only difference is that the icons are different in order to identify more easily the pins of a component. Usually, connector PositivePin is used for the positive and connector NegativePin for the negative pin of an electrical component.</p>
</html>", revisions = "<html>
<ul>
<li><i> 1998   </i>
       by Christoph Clauss<br> initially implemented<br>
       </li>
</ul>
</html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100,
          100}}), graphics = {Rectangle(
          extent = {{-100, 100}, {100, -100}},
          lineColor = {0, 0, 255},
          fillColor = {0, 0, 255},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0})), Protection(access = Access.icon));

    end PositivePin;
    connector NegativePin "负极接口"
      extends Pin;
      annotation (defaultComponentName = "pin_n",
        Documentation(info = "<html>
<p>Connectors PositivePin and NegativePin are nearly identical. The only difference is that the icons are different in order to identify more easily the pins of a component. Usually, connector PositivePin is used for the positive and connector NegativePin for the negative pin of an electrical component.</p>
</html>", revisions = "<html>
<dl>
<dt><i>1998</i></dt>
<dd>by Christoph Clauss initially implemented
</dd>
</dl>
</html>"),
        Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100,
          100}}), graphics = {Rectangle(
          extent = {{-100, 100}, {100, -100}},
          lineColor = {0, 0, 255},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0})), Protection(access = Access.icon));

    end NegativePin;
    annotation (Protection(access = Access.icon));
    import SI = Modelica.SIunits;
  end Electrical;
  package Power "功率接口"
    connector PowerPort "基础功率口"
      flow SI.Power P "接口传递功率";
      SI.AngularVelocity N(displayUnit = "rpm") "实际转速";
      stream Real C_exergy "成本流";
      stream SI.Power Exergy "㶲";
      // stream SI.SpecificEnergy exergy "比㶲";

      annotation (Protection(access = Access.icon));
    end PowerPort;
    connector PowerPort_a "功率接口a"
      extends PowerPort;
      annotation (
        defaultComponentName = "flange_a",
        Documentation(info = "<html>
<p>
This is a connector for 1-dim. rotational mechanical systems and models
the mechanical flange of a shaft. The following variables are defined in this connector:
</p>

<table border=1 cellspacing=0 cellpadding=2>
  <tr><td valign=\"top\"> <b>phi</b></td>
      <td valign=\"top\"> Absolute rotation angle of the shaft flange in [rad] </td>
  </tr>
  <tr><td valign=\"top\"> <b>tau</b></td>
      <td valign=\"top\"> Cut-torque in the shaft flange in [Nm] </td>
  </tr>
</table>

<p>
There is a second connector for flanges: Flange_b. The connectors
Flange_a and Flange_b are completely identical. There is only a difference
in the icons, in order to easier identify a flange variable in a diagram.
For a discussion on the actual direction of the cut-torque tau and
of the rotation angle, see section
<a href=\"modelica://Modelica.Mechanics.Rotational.UsersGuide.SignConventions\">Sign Conventions</a>
in the user's guide of Rotational.
</p>

<p>
If needed, the absolute angular velocity w and the
absolute angular acceleration a of the flange can be determined by
differentiation of the flange angle phi:
</p>
<pre>
     w = der(phi);    a = der(w)
</pre>
</html>"), Icon(coordinateSystem(preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}}), graphics = {
          Ellipse(
          extent = {{-100, 100}, {100, -100}},
          lineColor = {0, 0, 0},
          fillColor = {95, 95, 95},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(
          preserveAspectRatio = true,
          extent = {{-100, -100}, {100, 100}}), graphics = {Text(
          extent = {{-160, 90}, {40, 50}},
          lineColor = {0, 0, 0},
          textString = "%name"), Ellipse(
          extent = {{-40, 40}, {40, -40}},
          lineColor = {0, 0, 0},
          fillColor = {135, 135, 135},
          fillPattern = FillPattern.Solid)}), Protection(access = Access.icon));

    end PowerPort_a;

    connector PowerPort_b "功率接口b"
      extends PowerPort;
      annotation (
        defaultComponentName = "flange_b",
        Documentation(info = "<html>
<p>
This is a connector for 1-dim. rotational mechanical systems and models
the mechanical flange of a shaft. The following variables are defined in this connector:
</p>

<table border=1 cellspacing=0 cellpadding=2>
  <tr><td valign=\"top\"> <b>phi</b></td>
      <td valign=\"top\"> Absolute rotation angle of the shaft flange in [rad] </td>
  </tr>
  <tr><td valign=\"top\"> <b>tau</b></td>
      <td valign=\"top\"> Cut-torque in the shaft flange in [Nm] </td>
  </tr>
</table>

<p>
There is a second connector for flanges: Flange_a. The connectors
Flange_a and Flange_b are completely identical. There is only a difference
in the icons, in order to easier identify a flange variable in a diagram.
For a discussion on the actual direction of the cut-torque tau and
of the rotation angle, see section
<a href=\"modelica://Modelica.Mechanics.Rotational.UsersGuide.SignConventions\">Sign Conventions</a>
in the user's guide of Rotational.
</p>

<p>
If needed, the absolute angular velocity w and the
absolute angular acceleration a of the flange can be determined by
differentiation of the flange angle phi:
</p>
<pre>
     w = der(phi);    a = der(w)
</pre>
</html>"), Icon(coordinateSystem(
          preserveAspectRatio = true,
          extent = {{-100, -100}, {100, 100}}), graphics = {Ellipse(
          extent = {{-98, 100}, {102, -100}},
          lineColor = {0, 0, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid)}),
        Diagram(coordinateSystem(
          preserveAspectRatio = true,
          extent = {{-100, -100}, {100, 100}}), graphics = {Ellipse(
          extent = {{-40, 40}, {40, -40}},
          lineColor = {0, 0, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid), Text(
          extent = {{-40, 90}, {160, 50}},
          lineColor = {0, 0, 0},
          textString = "%name")}), Protection(access = Access.icon));

    end PowerPort_b;
    annotation (Protection(access = Access.icon));
    import SI = Modelica.SIunits;
  end Power;
  package ElectricPower "电功率接口"
    connector Electrical "电功率接口"
      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Polygon(origin = {2.0, 0.0},
        lineColor = {0, 85, 255},
        fillColor = {0, 85, 255},
        fillPattern = FillPattern.Solid,
        lineThickness = 0.5,
        points = {{-80.0, 50.0}, {80.0, 50.0}, {100.0, 30.0}, {80.0, -40.0}, {60.0, -50.0}, {-60.0, -50.0}, {-80.0, -40.0}, {-100.0, 30.0}},
        smooth = Smooth.Bezier)}), Protection(access = Access.icon));
      flow SI.Power P_plan "计划功率";
      flow SI.Power P_act "实际功率";
      flow Real C "现金流";
      Real Capex "投资";
      Real c1 "输入电价";
      Real c2 "输出电价";
    end Electrical;
    import SI = Modelica.SIunits;
    annotation (Protection(access = Access.icon));
  end ElectricPower;
  annotation (Protection(access = Access.icon));
end Interfaces;
