within TypicalScensrio;

package TypicalScenarios "典型场景模型"


  model Wind "风力发电出力模型"
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c = 0.3 "风电电价，单位：元/千瓦时";
    parameter Modelica.SIunits.Power Pn = 2e6 "风机装机容量" annotation (Dialog(group = "设备参数"));
    parameter Modelica.SIunits.Velocity vci = 3.5 "切入风速" annotation (Dialog(group = "设备参数"));
    parameter Modelica.SIunits.Velocity vco = 15 "切出风速" annotation (Dialog(group = "设备参数"));
    parameter Modelica.SIunits.Velocity vr = 10 "额定风速" annotation (Dialog(group = "设备参数"));
    parameter TypicalScensrio.Utilities.Types.Cost Capex = 3000 "造价，单位：元/千瓦时";

    Modelica.SIunits.Velocity v "实际风速";
    TypicalScensrio.Interfaces.ElectricPower.Electrical P_WT "风电功率" annotation (Placement(transformation(origin = {114, -16.000000000000004},
      extent = {{20, -20}, {-20, 20}},
      rotation = -180),
      iconTransformation(origin = {102, 0},
        extent = {{-9.000000000000014, -9.000000000000014}, {9.000000000000014, 9.000000000000014}},
        rotation = 270)));
    Modelica.Blocks.Interfaces.RealInput v_in "风速" 
      annotation (Placement(transformation(origin = {-100.0, -16.0},
        extent = {{20.0, -20.0}, {-20.0, 20.0}},
        rotation = -180.0),
        iconTransformation(origin = {-100.0, 71.99999999999997},
          extent = {{10.0, -10.0}, {-10.0, 9.999999999999993}},
          rotation = -180.0)));
    TypicalScensrio.Utilities.Types.CostFlow C "现金流";
    annotation (Icon(coordinateSystem(extent = {{-100, -100}, {100, 100}},
      grid = {2, 2}), graphics = {Rectangle(origin = {0, 0},
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      extent = {{-100, 100}, {100, -100}}), Ellipse(origin = {-10, -60},
      fillColor = {164, 198, 61},
      fillPattern = FillPattern.Solid,
      extent = {{-85, 25}, {85, -25}}), Polygon(origin = {-30, 27},
      fillColor = {94, 139, 172},
      fillPattern = FillPattern.Solid,
      points = {{-15, -12}, {-11, -8}, {15, 18}, {11, 12}, {5, 4}, {1, -2}, {-9, -18}}), Polygon(origin = {-73, 9},
      fillColor = {101, 146, 184},
      fillPattern = FillPattern.Solid,
      points = {{22, 2}, {10, 4}, {-22, 6}, {20, -6}}), Polygon(origin = {-38, -18},
      fillColor = {102, 149, 179},
      fillPattern = FillPattern.Solid,
      points = {{-11, 17}, {11, -21}, {-1, 21}}), Ellipse(origin = {-43.999999999999986, 5.999999999999986},
      fillColor = {129, 176, 53},
      fillPattern = FillPattern.Solid,
      extent = {{-9.000000000000014, 9.000000000000014}, {9.000000000000014, -9.000000000000014}}), Polygon(origin = {-45, -30},
      fillColor = {55, 82, 25},
      fillPattern = FillPattern.Solid,
      points = {{-4, 29}, {-4, -29}, {4, -29}, {0, 21}}), Polygon(origin = {50, 21},
      fillColor = {94, 139, 172},
      fillPattern = FillPattern.Solid,
      points = {{-15, -12}, {-11, -8}, {15, 18}, {11, 12}, {5, 4}, {1, -2}, {-9, -18}}), Polygon(origin = {7, 2},
      fillColor = {101, 146, 184},
      fillPattern = FillPattern.Solid,
      points = {{22, 3}, {10, 5}, {-22, 7}, {20, -7}}), Polygon(origin = {42, -24},
      fillColor = {102, 149, 179},
      fillPattern = FillPattern.Solid,
      points = {{-11, 17}, {11, -21}, {-1, 21}}), Ellipse(origin = {34, 2},
      fillColor = {129, 176, 53},
      fillPattern = FillPattern.Solid,
      extent = {{-7, 7}, {7, -7}}), Polygon(origin = {35, -36},
      fillColor = {55, 82, 25},
      fillPattern = FillPattern.Solid,
      points = {{-4, 29}, {-4, -29}, {4, -29}, {0, 21}})}), Protection(access = Access.icon));
  equation
    v = v_in;
    //风速低于切入风速时
    if v >= 0 and v < vci then
      P_WT.P_plan = 0;
      //风速高于切入风速，但低于额定风速时
    elseif v >= vci and v < vr then
      P_WT.P_plan + Pn * (v ^ 3 / (vr ^ 3 - vci ^ 3) - vci ^ 3 / (vr ^ 3 - vci ^ 3)) = 0;
      //风速高于额定风速，但低于切除风速时
    elseif v >= vr and v < vco then
      P_WT.P_plan + Pn = 0;
      //风速高于切出风速时
    else
      P_WT.P_plan = 0;
    end if;
    //单位换算
    P_WT.C = -P_WT.P_act * P_WT.c2 / 3.6e6;
    c = P_WT.c2;
    P_WT.c1 = 0;
    P_WT.C - C = 0;
    P_WT.Capex = Capex;
    annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
      grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, 0.0},
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      extent = {{-100.0, 100.0}, {100.0, -100.0}}), Ellipse(origin = {-5.0, -55.0},
      fillColor = {164, 198, 61},
      fillPattern = FillPattern.Solid,
      extent = {{-85.0, 25.0}, {85.0, -25.0}}), Polygon(origin = {-25.0, 32.0},
      fillColor = {94, 139, 172},
      fillPattern = FillPattern.Solid,
      points = {{-15.0, -12.0}, {-11.0, -8.0}, {15.0, 18.0}, {11.0, 12.0}, {5.0, 4.0}, {1.0, -2.0}, {-9.0, -18.0}}), Polygon(origin = {-68.0, 14.0},
      fillColor = {101, 146, 184},
      fillPattern = FillPattern.Solid,
      points = {{22.0, 2.0}, {10.0, 4.0}, {-22.0, 6.0}, {20.0, -6.0}}), Polygon(origin = {-33.0, -13.0},
      fillColor = {102, 149, 179},
      fillPattern = FillPattern.Solid,
      points = {{-11.0, 17.0}, {11.0, -21.0}, {-1.0, 21.0}}), Ellipse(origin = {-38.999999999999986, 10.999999999999986},
      fillColor = {129, 176, 53},
      fillPattern = FillPattern.Solid,
      extent = {{-9.000000000000014, 9.000000000000014}, {9.000000000000014, -9.000000000000014}}), Polygon(origin = {-40.0, -25.0},
      fillColor = {55, 82, 25},
      fillPattern = FillPattern.Solid,
      points = {{-4.0, 29.0}, {-4.0, -29.0}, {4.0, -29.0}, {0.0, 21.0}}), Polygon(origin = {55.0, 26.0},
      fillColor = {94, 139, 172},
      fillPattern = FillPattern.Solid,
      points = {{-15.0, -12.0}, {-11.0, -8.0}, {15.0, 18.0}, {11.0, 12.0}, {5.0, 4.0}, {1.0, -2.0}, {-9.0, -18.0}}), Polygon(origin = {12.0, 8.0},
      fillColor = {101, 146, 184},
      fillPattern = FillPattern.Solid,
      points = {{22.0, 2.0}, {10.0, 4.0}, {-22.0, 6.0}, {20.0, -6.0}}), Polygon(origin = {47.0, -19.0},
      fillColor = {102, 149, 179},
      fillPattern = FillPattern.Solid,
      points = {{-11.0, 17.0}, {11.0, -21.0}, {-1.0, 21.0}}), Ellipse(origin = {39.0, 7.0},
      fillColor = {129, 176, 53},
      fillPattern = FillPattern.Solid,
      extent = {{-7.0, 7.0}, {7.0, -7.0}}), Polygon(origin = {40.0, -31.0},
      fillColor = {55, 82, 25},
      fillPattern = FillPattern.Solid,
      points = {{-4.0, 29.0}, {-4.0, -29.0}, {4.0, -29.0}, {0.0, 21.0}}), Text(origin = {0.0, -145.0},
      lineColor = {0, 85, 255},
      extent = {{-145.0, 35.0}, {145.0, -35.0}},
      textString = "Wind Power",
      fontSize = 144,
      textStyle = {TextStyle.None},
      textColor = {0, 85, 255})}));
  end Wind;



  model PV_e "光伏发电"
    //--------典型参数--------
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c = 0.3 "光伏电价（上网电价或度电成本）";
    parameter SI.Power Pn = 2e6 "光伏装机容量" annotation (Dialog(group = "设备参数"));
    parameter Real eta = 0.95 "光电转化效率" annotation (Dialog(group = "设备参数"));
    //--------温度参数--------
    parameter SI.EnergyFluenceRate Gstc = 1000 annotation (Dialog(group = "环境参数"));
    parameter SI.Temperature T_stc = 298.15 "标准温度" annotation (Dialog(group = "环境参数"));
    parameter Real KT = 0.005 "温度系数" annotation (Dialog(group = "环境参数"));
    // parameter CAESS.Utilities.Types.ExergyCost_kWh price = 0.265 "上网电价";
    parameter TypicalScensrio.Utilities.Types.Cost Capex = 6e6 "造价";

    TypicalScensrio.Utilities.Types.CostFlow C "现金流";

    Modelica.Blocks.Interfaces.RealInput G_in "辐照强度，W/m2" 
      annotation (Placement(transformation(origin = {-100.0, -19.999999999999993},
        extent = {{20.0, -20.0}, {-20.0, 20.0}},
        rotation = -180.0),
        iconTransformation(origin = {-107.0, 64.99999999999997},
          extent = {{17.0, -17.0}, {-17.0, 16.999999999999993}},
          rotation = -180.0)));

    Modelica.Blocks.Interfaces.RealInput v_in "风速,m/S" 
      annotation (Placement(transformation(origin = {-100.0, 34.0},
        extent = {{20.0, -20.0}, {-20.0, 20.0}},
        rotation = -180.0),
        iconTransformation(origin = {-107.00000000000001, -3.0000000000000133},
          extent = {{16.999999999999986, -16.999999999999986}, {-16.999999999999986, 16.999999999999986}},
          rotation = -180.0)));

    Modelica.Blocks.Interfaces.RealInput T_air "环境温度" 
      annotation (Placement(transformation(origin = {-100.0, -73.99999999999999},
        extent = {{20.0, -20.0}, {-20.0, 20.0}},
        rotation = -180.0),
        iconTransformation(origin = {-107.00000000000001, -84.00000000000001},
          extent = {{16.999999999999986, -16.999999999999986}, {-16.999999999999986, 16.999999999999986}},
          rotation = -180.0)));
  protected
    SI.Temperature T_pv "光伏表面温度";
  public
    TypicalScensrio.Interfaces.ElectricPower.Electrical P_PV "光伏发电功率" annotation (Placement(transformation(origin = {102.00000000000001, 6.499999999999971},
      extent = {{10.0, -10.0}, {-10.0, 10.0}},
      rotation = -90.0)));
    import SI = Modelica.SIunits;
  equation
    P_PV.P_plan + max(Pn * G_in / Gstc * (1 - KT * (T_pv - T_stc)) * eta, 0) = 0;

    T_pv = T_air + 0.0138 * (1 + 0.031 * (T_air - 273.15)) * (1 - 0.042 * v_in) * G_in;

    P_PV.C = -P_PV.P_act * P_PV.c2 / 3.6e6;
    P_PV.C - C = 0;
    c = P_PV.c2;
    P_PV.c1 = 0;
    P_PV.Capex = Capex;
    annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
      grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, -0.5},
      fillColor = {146, 146, 146},
      fillPattern = FillPattern.Solid,
      extent = {{-100.0, 100.5}, {100.0, -100.5}}), Ellipse(origin = {-49.0, 59.0},
      lineColor = {255, 215, 0},
      fillColor = {255, 215, 0},
      fillPattern = FillPattern.Solid,
      extent = {{-17.189999999999998, 17.189999999999998}, {17.19, -17.189999999999998}}), Line(origin = {-73.0, 31.0},
      points = {{-5.73, -5.73}, {5.73, 5.73}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {-48.0, 25.0},
      points = {{0.0, -7.26}, {0.0, 7.26}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {-23.0, 31.0},
      points = {{5.73, -5.73}, {-5.73, 5.73}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {-80.0, 58.0},
      points = {{-7.64, 0.0}, {7.64, 0.0}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {-10.0, 58.0},
      points = {{-7.64, 0.0}, {7.64, 0.0}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {-75.0, 83.0},
      points = {{-5.73, 5.73}, {5.73, -5.73}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {-48.0, 91.0},
      points = {{0.0, 7.26}, {0.0, -7.26}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {-17.0, 83.0},
      points = {{5.73, 5.73}, {-5.73, -5.73}},
      color = {255, 255, 0},
      thickness = 1.0), Line(origin = {9.0, 31.0},
      points = {{-11.0, 9.0}, {11.0, -9.0}},
      color = {255, 255, 0},
      thickness = 3.0,
      arrow = {Arrow.None, Arrow.Filled}), Line(origin = {45.0, 33.0},
      points = {{-11.0, 9.0}, {11.0, -9.0}},
      color = {255, 255, 0},
      thickness = 3.0,
      arrow = {Arrow.None, Arrow.Filled}), Text(origin = {0.0, -135.0},
      lineColor = {0, 85, 255},
      extent = {{-140.0, 35.0}, {140.0, -35.0}},
      textString = "Photovoltaic",
      fontSize = 144,
      textStyle = {TextStyle.None},
      textColor = {0, 85, 255}), Rectangle(origin = {-82.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-62.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-42.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-82.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-62.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-42.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-82.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-62.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-42.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-18.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {2.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {22.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-18.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {2.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {22.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {-18.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {2.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {22.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {48.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {68.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {88.0, -10.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {48.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {68.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {88.0, -30.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {48.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {68.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}}), Rectangle(origin = {88.0, -50.0},
      lineColor = {255, 255, 255},
      fillColor = {0, 0, 128},
      fillPattern = FillPattern.Solid,
      extent = {{-10.0, 10.0}, {10.0, -10.0}})}), Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0})), Protection(access = Access.icon));
  end PV_e;
  model ThermalPower
    // parameter SI.Time deltaT = 3600;
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c = 0.3 "电价（上网电价或度电成本）";

    parameter Real table[:,2] = {{3600, 1}, {7200, 1}, {10800, 1}, {14400, 1}, {18000, 1}, {21600, 1}, {25200, 1}, {28800, 1}, {32400, 1}, {36000, 1}, {39600, 1}, {43200, 1}, {46800, 1}, {50400, 1}, {54000, 1}, {57600, 1}, {61200, 1}, {64800, 1}, {68400, 1}, {72000, 1}, {75600, 1}, {79200, 1}, {82800, 1}, {86400, 1}} "负荷率";
    parameter Modelica.SIunits.Power P_cap = 6e8 "装机容量";
    parameter Modelica.SIunits.Power P_max = 6e8 "最大出力";
    parameter Modelica.SIunits.Power P_min = 1.8e8 "最小出力";
    parameter Modelica.SIunits.TimeAging rate_max = 0.0025 / 60 "最大变负荷速率";

    parameter TypicalScensrio.Utilities.Types.Cost Capex = 3e9 "造价";
    parameter Real k = 1 "罚函数系数";

    Real C_penality "罚函数";

    // SI.TimeAging rate "负荷变化速率";
    // SI.Power P_act0(start = -table[1,2] * P_cap);
    TypicalScensrio.Interfaces.ElectricPower.Electrical positivePlug annotation (Placement(transformation(origin = {102.0, 2.513069627573458},
      extent = {{10.0, -10.0}, {-10.0, 10.0}},
      rotation = -90.0)));
    Modelica.Blocks.Sources.CombiTimeTable Table(table = table,
      extrapolation = Modelica.Blocks.Types.Extrapolation.LastTwoPoints,
      smoothness = Modelica.Blocks.Types.Smoothness.ConstantSegments) 
      annotation (Placement(transformation(origin = {-1.0658141036401503e-14, 2.0},
        extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    Modelica.Blocks.Interfaces.RealInput u_dispatch(start = 1) "外部火电负荷率调度指令" annotation (Placement(transformation(origin = {-104.0, 2.0},
      extent = {{-14.0, -14.0}, {14.0, 14.0}})));

    TypicalScensrio.Utilities.Types.CostFlow C "现金流";
    annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
      grid = {2.0, 2.0}), graphics = {Rectangle(origin = {1.6580673374971298, 2.5130696275734508},
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      lineThickness = 0.5,
      extent = {{-83.83811586860976, 68.17413832986355}, {83.83811586860976, -68.17413832986355}}), Bitmap(origin = {1.6580673374971333, 2.5130696275734543},
      extent = {{-100.34466608695038, -85.16977151763646}, {100.34466608695037, 85.16977151763648}},
      fileName = "F:/Works-2023/projects/知识产权/Modeilica大会/论文/3.png"), Text(origin = {1.6580673374971386, -138.4235807860262},
      lineColor = {0, 85, 255},
      extent = {{-140.0, 35.0}, {140.0, -35.0}},
      textString = "火电",
      textStyle = {TextStyle.None},
      textColor = {0, 85, 255}), Rectangle(origin = {1.6580673374971227, 2.513069627573465},
      fillColor = {255, 255, 255},
      lineThickness = 2.0,
      extent = {{-100.34466608695038, 85.16977151763646}, {100.34466608695038, -85.16977151763646}})}),
      Protection(access = Access.icon));
    // SI.Power P_act
  equation
    positivePlug.P_plan = -u_dispatch * P_cap;
    // positivePlug.C = -positivePlug.P_act * positivePlug.c2 / 3.6e6 + k * max(positivePlug.P_act + P_min, 0) - k * min(P_max + positivePlug.P_act, 0);
    positivePlug.C = -positivePlug.P_act * positivePlug.c2 / 3.6e6 + C_penality;
    // C_penality = k * P_cap * (e ^ max(positivePlug.P_act / P_cap + P_min / P_cap, 0) - 1) + k * (e ^ (-min(P_max / P_cap + positivePlug.P_act / P_cap, 0)) - 1);
    C_penality = k * P_cap * (e ^ max(positivePlug.P_act + P_min, 0) - 1) + k * (e ^ (-min(P_max + positivePlug.P_act, 0)) - 1);

    positivePlug.C - C = 0;
    positivePlug.Capex = Capex;
    // positivePlug.P_act = -min(min(max(max(-positivePlug.P_plan, P_min), -P_act0 - (rate_max * P_cap * deltaT)), -P_act0 + (rate_max * P_cap * deltaT)), P_max);
    positivePlug.P_act = positivePlug.P_plan;
    c = positivePlug.c2;
    positivePlug.c1 = 0;
    // when sample(0, deltaT) then 
    //   P_act0 = pre(positivePlug.P_act);
    // end when;
    // P_act0 = pre(positivePlug.P_act);
  end ThermalPower;
  model ELoad "电负荷"
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c = 0.3 "用电电价";
    parameter Real table[:,2] = {{3600, 1}, {7200, 1}, {10800, 1}, {14400, 1}, {18000, 1}, {21600, 1}, {25200, 1}, {28800, -0.1}, {32400, -0.3}, {36000, -0.4}, {39600, -0.4}, {43200, -0.4}, {46800, -0.25}, {50400, -0.31}, {54000, 1}, {57600, 1}, {61200, -0.4}, {64800, -0.4}, {68400, -0.8}, {72000, -0.5}, {75600, 0.6}, {79200, 0.8}, {82800, 0.5}, {86400, 0.7}};
    parameter TypicalScensrio.Utilities.Types.Cost Capex = 0 "造价";
    TypicalScensrio.Interfaces.ElectricPower.Electrical ELoad annotation (Placement(transformation(origin = {102.0, 2.513069627573458},
      extent = {{10.0, -10.0}, {-10.0, 10.0}},
      rotation = -90.0)));
    Modelica.Blocks.Sources.CombiTimeTable Table(table = table,
      extrapolation = Modelica.Blocks.Types.Extrapolation.LastTwoPoints,
      smoothness = Modelica.Blocks.Types.Smoothness.ConstantSegments) 
      annotation (Placement(transformation(origin = {-1.0658141036401503e-14, 2.0},
        extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    TypicalScensrio.Utilities.Types.CostFlow C "现金流";
  equation
    -ELoad.P_plan + Table.y[1] = 0;
    ELoad.C = -ELoad.P_act * ELoad.c1 / 3.6e6;
    ELoad.C + C = 0;
    ELoad.Capex = Capex;
    c = ELoad.c1;
    ELoad.c2 = 0;
    annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
      grid = {2.0, 2.0}), graphics = {Text(origin = {0.0, -145.0},
      lineColor = {0, 85, 255},
      extent = {{-140.0, 35.0}, {140.0, -35.0}},
      textString = "电负荷",
      textStyle = {TextStyle.None},
      textColor = {0, 85, 255}), Rectangle(origin = {0.0, 0.0},
      lineColor = {0, 85, 255},
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      lineThickness = 1.0,
      extent = {{-100.0, 100.0}, {100.0, -100.0}}), Line(origin = {-60.0, 0.0},
      rotation = 90.0,
      points = {{-100.0, 0.0}, {100.0, 0.0}},
      color = {0, 0, 0},
      thickness = 0.5,
      smooth = Smooth.Bezier), Line(origin = {60.0, 0.0},
      rotation = -90.0,
      points = {{-100.0, 0.0}, {100.0, 0.0}},
      color = {0, 0, 0},
      thickness = 0.5,
      smooth = Smooth.Bezier), Line(origin = {0.0, 10.0},
      rotation = -90.0,
      points = {{10.0, 80.0}, {-10.0, 60.0}, {10.0, 40.0}, {-10.0, 20.0}, {10.0, 0.0}, {-10.0, -20.0}, {10.0, -40.0}, {-10.0, -60.0}, {10.0, -80.0}},
      color = {0, 0, 255},
      thickness = 0.5,
      smooth = Smooth.Bezier), Line(origin = {0.0, -30.0},
      rotation = 270.0,
      points = {{10.0, 80.0}, {-10.0, 60.0}, {10.0, 40.0}, {-10.0, 20.0}, {10.0, 0.0}, {-10.0, -20.0}, {10.0, -40.0}, {-10.0, -60.0}, {10.0, -80.0}},
      color = {0, 0, 255},
      thickness = 0.5,
      smooth = Smooth.Bezier), Line(origin = {0.0, 50.0},
      rotation = -90.0,
      points = {{10.0, 80.0}, {-10.0, 60.0}, {10.0, 40.0}, {-10.0, 20.0}, {10.0, 0.0}, {-10.0, -20.0}, {10.0, -40.0}, {-10.0, -60.0}, {10.0, -80.0}},
      color = {0, 0, 255},
      thickness = 0.5,
      smooth = Smooth.Bezier), Polygon(origin = {80.5, 39.5},
      rotation = -90.0,
      lineColor = {0, 0, 255},
      fillColor = {0, 0, 255},
      fillPattern = FillPattern.Solid,
      lineThickness = 0.5,
      points = {{2.5, -5.5}, {5.5, 5.5}, {-5.5, 2.5}, {2.5, -5.5}}), Polygon(origin = {80.0, -1.0},
      rotation = -90.0,
      lineColor = {0, 0, 255},
      fillColor = {0, 0, 255},
      fillPattern = FillPattern.Solid,
      lineThickness = 0.5,
      points = {{2.0, -5.0}, {6.0, 5.0}, {-6.0, 3.0}, {2.0, -5.0}}), Polygon(origin = {82.0, -41.0},
      rotation = -90.0,
      lineColor = {0, 0, 255},
      fillColor = {0, 0, 255},
      fillPattern = FillPattern.Solid,
      lineThickness = 0.5,
      points = {{2.0, -5.0}, {6.0, 5.0}, {-6.0, 3.0}, {2.0, -5.0}})}), Protection(access = Access.icon));
  end ELoad;

  model Grid "电网能源"
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c_sale = 0.3 "上网电价";
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c_buy = 0.2 "购电电价";
    parameter SI.Power P1(displayUnit = "MW") = 1e7 "最大上网功率";
    parameter SI.Power P2 = -10 "最大下网功率";
    parameter TypicalScensrio.Utilities.Types.Cost Capex = 0 "造价";
    // parameter CAESS.Utilities.Types.ExergyCost_kWh c1 = 0.2 "惩罚购电";
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power 
      annotation (Placement(transformation(origin = {-99.99999999999999, -6.661338147750939e-16},
        extent = {{-10.0, -10.0}, {10.0, 10.0}},
        rotation = 90.0)));
    import SI = Modelica.SIunits;
  equation
    if Power.P_plan > 0 then
      Power.P_act = min(Power.P_plan, P1);
      Power.C = -Power.c1 * Power.P_act / 3.6e6;
    else
      Power.P_act = max(Power.P_plan, P2);
      Power.C = -Power.c2 * Power.P_act / 3.6e6;
    end if;
    Power.c1 = c_sale;
    Power.c2 = c_buy;
    Power.Capex = Capex;
    annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
      grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, 0.0},
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      extent = {{-100.0, 100.0}, {100.0, -100.0}}), Rectangle(origin = {0.0, -80.0},
      fillColor = {78, 173, 14},
      fillPattern = FillPattern.Solid,
      extent = {{-100.0, 20.0}, {100.0, -20.0}}), Line(origin = {-39.0, 35.0},
      points = {{17.0, 33.0}, {-17.0, -1.0}, {13.0, -1.0}, {-15.0, -33.0}, {15.0, -33.0}},
      thickness = 2.0), Line(origin = {38.0, 35.0},
      points = {{-20.0, 33.0}, {14.0, -1.0}, {-14.0, -1.0}, {20.0, -33.0}, {-12.0, -33.0}},
      thickness = 2.0), Line(origin = {1.0, -37.0},
      points = {{-21.0, 39.0}, {-35.0, -39.0}, {29.0, -7.0}, {35.0, -39.0}, {-29.0, -7.0}, {23.0, 17.0}, {-17.0, 39.0}, {17.0, 39.0}, {-23.0, 19.0}},
      thickness = 2.0), Line(origin = {3.0, -31.0},
      points = {{-27.0, 13.0}, {27.0, -13.0}},
      thickness = 2.0), Line(origin = {25.0, -21.0},
      points = {{-5.0, 23.0}, {5.0, -23.0}},
      thickness = 2.0), Line(origin = {-1.0, -78.0},
      points = {{-57.0, 0.0}, {57.0, 0.0}},
      thickness = 2.0), Line(origin = {52.0, 31.0},
      points = {{0.0, 3.0}, {0.0, -3.0}},
      thickness = 2.0), Line(origin = {58.0, -1.0},
      points = {{0.0, 3.0}, {0.0, -3.0}},
      thickness = 2.0), Line(origin = {-2.0, 18.0},
      points = {{-18.0, -16.0}, {18.0, 16.0}},
      thickness = 2.0), Line(origin = {1.0, 18.0},
      points = {{19.0, -16.0}, {-19.0, 16.0}},
      thickness = 2.0), Line(origin = {-1.0, 34.0},
      points = {{-17.0, 0.0}, {17.0, 0.0}},
      thickness = 2.0), Line(origin = {-19.0, 18.0},
      points = {{1.0, 16.0}, {-1.0, -16.0}},
      thickness = 2.0), Line(origin = {18.0, 18.0},
      points = {{-2.0, 16.0}, {2.0, -16.0}},
      thickness = 2.0), Line(origin = {-3.0, 48.0},
      points = {{-15.0, -14.0}, {15.0, 14.0}},
      thickness = 2.0), Line(origin = {0.0, 48.0},
      points = {{16.0, -14.0}, {-16.0, 14.0}},
      thickness = 2.0), Line(origin = {-2.0, 62.0},
      points = {{-14.0, 0.0}, {14.0, 0.0}},
      thickness = 2.0), Line(origin = {-17.0, 49.0},
      points = {{1.0, 13.0}, {-1.0, -13.0}},
      thickness = 2.0), Line(origin = {14.0, 48.0},
      points = {{-2.0, 14.0}, {2.0, -14.0}},
      thickness = 2.0), Line(origin = {-69.0, 14.0},
      points = {{23.0, 18.0}, {-23.0, -18.0}},
      pattern = LinePattern.Dash), Line(origin = {-63.0, -12.0},
      points = {{23.0, 18.0}, {-23.0, -18.0}},
      pattern = LinePattern.Dash), Line(origin = {49.0, 24.0},
      points = {{23.0, 18.0}, {-23.0, -18.0}},
      pattern = LinePattern.Dash), Line(origin = {55.0, -8.0},
      points = {{23.0, 18.0}, {-23.0, -18.0}},
      pattern = LinePattern.Dash), Line(origin = {-2.0, 68.0},
      points = {{-20.0, 0.0}, {20.0, 0.0}},
      thickness = 2.0), Line(origin = {-2.0, 78.0},
      points = {{-20.0, -10.0}, {-6.0, 10.0}, {10.0, 10.0}, {20.0, -10.0}},
      thickness = 2.0), Text(origin = {0.0, -145.0},
      lineColor = {0, 85, 255},
      extent = {{-140.0, 35.0}, {140.0, -35.0}},
      textString = "电网能源",
      textStyle = {TextStyle.None},
      textColor = {0, 85, 255})}), Protection(access = Access.icon));
  end Grid;

  model Battery "蓄能电池"
    // parameter SI.Time deltaT = 3600;
    parameter Real table[:,2] = {{3600, 1}, {7200, 1}, {10800, 1}, {14400, 1}, {18000, 1}, {21600, 1}, {25200, 1}, {28800, -0.1}, {32400, -0.3}, {36000, -0.4}, {39600, -0.4}, {43200, -0.4}, {46800, -0.25}, {50400, -0.31}, {54000, 1}, {57600, 1}, {61200, -0.4}, {64800, -0.4}, {68400, -0.8}, {72000, -0.5}, {75600, 0.6}, {79200, 0.8}, {82800, 0.5}, {86400, 0.7}};
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c_buy = 0.2 "购电成本，单位：元/千瓦时";
    parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c_sale = 0.3 "售电成本，单位：元/千瓦时";
    parameter Modelica.SIunits.Power P_cap(displayUnit = "MW") = 1e8 "功率装机容量";
    parameter Modelica.SIunits.Energy E_cap = 400e6 * 3600 "储能装机容量";
    parameter Real SOC_max = 0.95 "储能上限";
    parameter Real SOC_min = 0.1 "储能下限";
    parameter Real Capex = 1e7 "造价，单位：元/千瓦时";
    parameter Real SOC_start = 0.5;
    parameter Real eta = 0.85 "电池充放效率";
    parameter Real k = 1 "SOC罚函数系数";
    parameter TypicalScensrio.Utilities.Types.Cost Income_start = 0;
    TypicalScensrio.Utilities.Types.Cost Income(start = Income_start);
    Real C_penality "罚函数";
    Real SOC(start = SOC_start);
    // Real SOC0(start = SOC_start);
    // SI.Power P "充放电功率";
    TypicalScensrio.Interfaces.ElectricPower.Electrical PBS "接口" annotation (Placement(transformation(origin = {-86.0, 4.0000000000000036},
      extent = {{10.0, -10.0}, {-10.0, 10.0}},
      rotation = 90.0)));
    // Electrical PBS2 "负载接口" annotation (Placement(transformation(origin = {88.0, 3.500000000000002}, 
      // extent = {{-10.0, -10.0}, {10.0, 10.0}}, 
      // rotation = -90.0)));
    Modelica.Blocks.Sources.CombiTimeTable Table(table = table) 

      annotation (Placement(transformation(origin = {-1.0658141036401503e-14, 2.0},
        extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    Modelica.Blocks.Interfaces.RealInput u_dispatch(start = 0) "外部电池调度指令" annotation (Placement(transformation(origin = {-116.0, 4.0},
      extent = {{-14.0, -14.0}, {14.0, 14.0}})));
  equation
    PBS.P_plan = u_dispatch * P_cap;
    if PBS.P_act >= 0 then
      // PBS.C = -PBS.P_act * PBS.c1 / 3.6e6 - k * min(SOC - SOC_min, 0) - k * min(SOC_max - SOC, 0);
      PBS.C = -PBS.P_act * PBS.c1 / 3.6e6 + C_penality;
      C_penality = k * E_cap * (e ^ (-min(SOC - SOC_min, 0)) - 1) + k * (e ^ (-min(SOC_max - SOC, 0)) - 1);

      der(SOC) = PBS.P_act / E_cap;
    else
      // PBS.C = -PBS.P_act * PBS.c2 / 3.6e6 - k * min(SOC - SOC_min, 0) - k * min(SOC_max - SOC, 0);
      PBS.C = -PBS.P_act * PBS.c2 / 3.6e6 + C_penality;
      C_penality = k * E_cap * (e ^ (-min(SOC - SOC_min, 0)) - 1) + k * (e ^ (-min(SOC_max - SOC, 0)) - 1);
      der(SOC) = PBS.P_act / E_cap / eta;
    end if;
    PBS.P_act = PBS.P_plan;

    PBS.Capex = Capex;
    PBS.c2 = c_sale;
    PBS.c1 = c_buy;

    der(Income) = PBS.C;
    annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
      grid = {2.0, 2.0}), graphics = {Rectangle(origin = {1.0, 3.0},
      fillColor = {236, 236, 236},
      fillPattern = FillPattern.Solid,
      extent = {{-83.0, 81.0}, {83.0, -81.0}}), Text(origin = {6.0, -135.0},
      lineColor = {0, 85, 255},
      extent = {{-145.0, 35.0}, {145.0, -35.0}},
      textString = "%name",
      textStyle = {TextStyle.None},
      textColor = {0, 85, 255}), Rectangle(origin = {-1.0, 92.0},
      fillColor = {51, 76, 90},
      fillPattern = FillPattern.Solid,
      extent = {{-89.0, 10.0}, {89.0, -10.0}}), Rectangle(origin = {-70.0, 106.0},
      fillColor = {56, 130, 114},
      fillPattern = FillPattern.Solid,
      extent = {{-8.0, 4.0}, {8.0, -4.0}}), Rectangle(origin = {-46.0, 106.0},
      fillColor = {53, 126, 111},
      fillPattern = FillPattern.Solid,
      extent = {{-8.0, 4.0}, {8.0, -4.0}}), Rectangle(origin = {44.0, 106.0},
      fillColor = {52, 126, 110},
      fillPattern = FillPattern.Solid,
      extent = {{-8.0, 4.0}, {8.0, -4.0}}), Rectangle(origin = {68.0, 106.0},
      fillColor = {54, 128, 108},
      fillPattern = FillPattern.Solid,
      extent = {{-8.0, 4.0}, {8.0, -4.0}}), Rectangle(origin = {1.0, -88.0},
      fillColor = {51, 76, 90},
      fillPattern = FillPattern.Solid,
      extent = {{-89.0, 10.0}, {89.0, -10.0}}), Ellipse(origin = {-38.0, 1.0},
      fillColor = {79, 161, 220},
      fillPattern = FillPattern.Solid,
      extent = {{-18.0, 21.0}, {18.0, -21.0}}), Ellipse(origin = {48.0, 1.0},
      fillColor = {78, 160, 219},
      fillPattern = FillPattern.Solid,
      extent = {{-18.0, 21.0}, {18.0, -21.0}}), Rectangle(origin = {-37.0, 1.0},
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      extent = {{-11.0, 1.0}, {11.0, -1.0}}), Rectangle(origin = {49.0, 1.0},
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      extent = {{-11.0, 1.0}, {11.0, -1.0}}), Rectangle(origin = {-37.0, 1.0},
      rotation = 90.0,
      fillColor = {255, 255, 255},
      fillPattern = FillPattern.Solid,
      extent = {{-11.0, 1.0}, {11.0, -1.0}}), Polygon(origin = {4.0, -5.0},
      fillColor = {205, 82, 71},
      fillPattern = FillPattern.Solid,
      points = {{-14.0, -3.0}, {-6.0, 29.0}, {8.0, 29.0}, {0.0, 3.0}, {14.0, 3.0}, {-8.0, -29.0}, {-2.0, -3.0}})}), Protection(access = Access.icon));
  end Battery;
  model Bus "功率母线"
    parameter TypicalScensrio.Utilities.Types.Cost Income_start = 0;
    parameter Real k = 1 "电力系统惩罚因子调节系数";
    TypicalScensrio.Utilities.Types.Cost Income(start = Income_start);
    TypicalScensrio.Utilities.Types.Cost Capex "造价";
    Modelica.SIunits.Power P_res "功率偏差，缺口或弃电";
    Modelica.SIunits.Power P_res1 "功率偏差，缺口或弃电";
    Modelica.SIunits.Power P_res2 "功率偏差，缺口或弃电";
    Real OPT_goal "优化目标";
    Real C_penality "罚函数";

    annotation (Icon(coordinateSystem(extent = {{-100.0, -300.0}, {100.0, 300.0}},
      grid = {2.0, 2.0}), graphics = {Rectangle(origin = {6.0, 26.0},
      fillColor = {0, 0, 0},
      fillPattern = FillPattern.Solid,
      extent = {{-22.0, 255.0}, {22.0, -255.0}})}), Protection(access = Access.icon));
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power_PV 
      annotation (Placement(transformation(origin = {-26.000000000000025, 154.0},
        extent = {{-22.0, -19.0}, {22.0, 19.0}},
        rotation = 90.0)));
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power_WT 
      annotation (Placement(transformation(origin = {-26.00000000000003, 227.99999999999994},
        extent = {{-22.0, -19.000000000000014}, {22.0, 19.0}},
        rotation = 90.0)));
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power_TP 
      annotation (Placement(transformation(origin = {-26.00000000000003, 80.00000000000003},
        extent = {{-22.0, -19.0}, {22.0, 18.999999999999996}},
        rotation = 90.0)));
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power_BT 
      annotation (Placement(transformation(origin = {-26.000000000000025, 8.000000000000053},
        extent = {{-22.0, -19.000000000000007}, {22.0, 18.999999999999993}},
        rotation = 90.0)));
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power_CAES 
      annotation (Placement(transformation(origin = {-26.00000000000003, -214.0000000000001},
        extent = {{-22.0, -19.000000000000007}, {22.0, 18.999999999999993}},
        rotation = 90.0)));
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power_Eload 
      annotation (Placement(transformation(origin = {-26.000000000000025, -140.00000000000006},
        extent = {{-21.999999999999964, -19.0}, {22.000000000000036, 19.0}},
        rotation = 90.0)));
    TypicalScensrio.Interfaces.ElectricPower.Electrical Power_Gird 
      annotation (Placement(transformation(origin = {-26.00000000000002, -66.0},
        extent = {{-22.0, -19.0}, {22.0, 18.999999999999996}},
        rotation = 90.0)));
  equation
    Power_PV.P_plan + Power_WT.P_plan + Power_TP.P_plan + Power_BT.P_plan + Power_CAES.P_plan + Power_Eload.P_plan + Power_Gird.P_plan = 0;
    0 = Power_PV.P_plan + Power_WT.P_plan + Power_TP.P_act + Power_BT.P_act + Power_CAES.P_act + Power_Eload.P_plan + Power_Gird.P_act + P_res1;

    // 策略，优先弃风电
    if P_res1 < 0 then //优先弃风电

      Power_WT.P_act = max((Power_WT.P_plan + P_res1), 0);//弃风电
      if Power_WT.P_act > 0 then
        P_res2 = 0;
      else
        P_res2 = -Power_WT.P_act + P_res1 + Power_WT.P_plan; //光伏的预计弃电量
      end if;
      Power_PV.P_act = max((Power_PV.P_plan + P_res2), 0);//再弃光伏
      P_res = -Power_PV.P_act + P_res2 + Power_PV.P_plan; // 光伏的实际弃电，负值代表弃电
      Power_Eload.P_plan = Power_Eload.P_act;
    else
      //用户降负荷
      Power_Eload.P_act = min((Power_Eload.P_plan + P_res1), 0.2 * Power_Eload.P_plan);
      if Power_Eload.P_act >= 0.2 * Power_Eload.P_plan then
        P_res2 = 0;
        P_res = Power_Eload.P_plan - Power_Eload.P_act + P_res1;// 电力缺口，正值代表负荷缺口
      else
        P_res2 = 0;
        P_res = 0;
      end if;
      Power_PV.P_act = Power_PV.P_plan;
      Power_WT.P_plan = Power_WT.P_act;
    end if;

    OPT_goal = Income;

    // 造价
    Capex = Power_PV.Capex + Power_WT.Capex + Power_TP.Capex + Power_BT.Capex + Power_CAES.Capex + Power_Eload.Capex + Power_Gird.Capex;
    der(Income) = (Power_PV.C + Power_WT.C + Power_TP.C + Power_BT.C + Power_CAES.C + Power_Eload.C + Power_Gird.C) - C_penality;
    C_penality = (k * P_res) ^ 2;

  end Bus;
  package CompressedAirEnergyStorage "压缩空气储能系统数据模型"
    model CompressedAirEnergyStorage "压缩空气储能系统数据模型"
      // parameter SI.Time deltaT = 3600;
      parameter Real table[:,2] = {{0, 0}, {3600, 0}, {3600, 0}, {7200, 0}, {7200, 0}, {10800, 0}, {10800, 0}, {14400, 0}, {14400, 0}, {18000, 0}, {18000, 0}, {21600, 0}, {21600, 0}, {25200, 0}, {25200, 0}, {28800, 0}} annotation (Dialog(group = "调度指令"));
      parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c_buy = 0.2 "购电成本，单位：元/千瓦时";
      parameter TypicalScensrio.Utilities.Types.ExergyCost_kWh c_sale = 0.3 "售电成本，单位：元/千瓦时";
      parameter Modelica.SIunits.Power P_cap(displayUnit = "MW") = 1.5e8 "功率装机容量";
      parameter Modelica.SIunits.Energy E_cap = 400e6 * 3600 "储能装机容量";
      parameter Real Capex = 1e7 "造价，单位：元/千瓦时";
      parameter Real k = 1 "SOC罚函数系数";
      parameter TypicalScensrio.Utilities.Types.Cost Income_start = 0;
      parameter Modelica.SIunits.Length level_coldtank_start = 30;
      parameter Modelica.SIunits.Length level_hottank_start = 30;
      parameter Modelica.SIunits.Pressure p_gastank_start = 85e5;
      parameter Modelica.SIunits.SpecificEnthalpy h_coldtank_start = 168952;
      parameter Modelica.SIunits.SpecificEnthalpy h_hottank_start = 852411;
      parameter Modelica.SIunits.SpecificEnthalpy h_gastank_start = 276700;


      Real C_GasTank_penality "罚函数";
      Real C_HotTank_penality "罚函数";
      Real C_ColdTank_penality "罚函数";
      TypicalScensrio.Utilities.Types.Cost Income(start = Income_start);
      Modelica.Blocks.Sources.CombiTimeTable Table(table = table,
        extrapolation = Modelica.Blocks.Types.Extrapolation.LastTwoPoints,
        smoothness = Modelica.Blocks.Types.Smoothness.ConstantSegments) annotation (Placement(transformation(origin = {-148.0, 41.39999999999999},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealInput u_dispatch(start = 0) "外部压缩空气储能调度指令" annotation (Placement(transformation(origin = {-148.0, 8.0},
        extent = {{-14.0, -14.0}, {14.0, 14.0}})));

      TypicalScensrio.Interfaces.ElectricPower.Electrical PBS "接口" annotation (Placement(transformation(origin = {105.99999999999999, -7.999999999999993},
        extent = {{10.0, -10.0}, {-10.0, 10.0}},
        rotation = 90.0)));

      Tank hottank(level_start = level_hottank_start,
        V0 = 5000 * 3,
        h_start = h_hottank_start) 
        annotation (Placement(transformation(origin = {34.0, 68.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Tank coldtank(T_start = 313.15, level_start = level_coldtank_start,
        V0 = 5000 * 3,
        h_start = h_coldtank_start) 


        annotation (Placement(transformation(origin = {34.0, 22.999999999999993},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_h1(redeclare package Medium = Media.FluidMedia.Water,
        use_mflow_in = true, use_T_in = true) annotation (Placement(transformation(origin = {-10.000000000000043, 68.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_h2(redeclare package Medium = Media.FluidMedia.Water,
        m_flow = 0,
        use_mflow_in = true) 
        annotation (Placement(transformation(origin = {78.0, 68.0},
          extent = {{10.0, -10.0}, {-10.0, 10.0}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_c2(redeclare package Medium = Media.FluidMedia.Water,
        m_flow = 0,
        use_mflow_in = true) 


        annotation (Placement(transformation(origin = {78.0, 22.99999999999997},
          extent = {{10.0, -10.0}, {-10.0, 10.0}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_c1(redeclare package Medium = Media.FluidMedia.Water,
        use_mflow_in = true) annotation (Placement(transformation(origin = {-10.000000000000043, 22.999999999999986},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      GasTank gastank(redeclare package Medium = Media.FluidMedia.IdealAir,
        p_start = p_gastank_start,
        h_start = h_gastank_start) 
        annotation (Placement(transformation(origin = {34.0000000000001, -68.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_g1(use_mflow_in = true,
        redeclare package Medium = Media.FluidMedia.IdealAir) annotation (Placement(transformation(origin = {-9.999999999999929, -68.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_g2(m_flow = 0,
        use_mflow_in = true, redeclare package Medium = Media.FluidMedia.IdealAir) annotation (Placement(transformation(origin = {82.00000000000006, -68.0},
          extent = {{10.0, -10.0}, {-10.0, 10.0}})));
      Utilities.Interpolations.Interpolation_3D mflow_hottank(ux = {258.45, 280.65, 301.05}, nx1 = 3, nx2 = 5, nx3 = 3, table = {{{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 146.5, 143.248, 140.775, 138.517, 136.648}, {140000000, 162.635, 161.035, 157.64, 155.494, 152.035}, {150000000, 180.028, 177.846, 173.701, 170.903, 169.002}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 141.183, 138.56, 136.369, 133.951, 132.245}, {140000000, 158.131, 155.37, 152.876, 150.628, 147.527}, {150000000, 175.585, 172.055, 169.349, 166.289, 163.654}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 137.262, 134.846, 132.417, 130.75, 128.724}, {140000000, 154.456, 151.168, 147.828, 145.605, 144.165}, {150000000, 169.74, 166.643, 164.308, 162.434, 159.555}}}) 
        annotation (Placement(transformation(origin = {-72.00000000000003, 71.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Utilities.Interpolations.Interpolation_3D T_hottank(ux = {258.45, 280.65, 301.05}, nx1 = 3, nx2 = 5, nx3 = 3, table = {{{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 450.625, 454.48, 456.317, 455.795, 453.532}, {140000000, 449.686, 449.195, 454.992, 453.455, 457.182}, {150000000, 452.505, 451.684, 455.468, 456.074, 453.318}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 464.411, 466.302, 464.169, 466.25, 469.374}, {140000000, 465.939, 467.027, 466.908, 463.663, 469.296}, {150000000, 462.528, 465.164, 466.159, 466.798, 467.542}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 474.929, 475.954, 474.866, 479.788, 480.021}, {140000000, 473.657, 476.282, 479.101, 480.453, 479.581}, {150000000, 476.377, 478.021, 477.284, 475.765, 477.6}}}) 
        annotation (Placement(transformation(origin = {-72.00000000000003, 106.99999999999997},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Utilities.Interpolations.Interpolation_3D mflow_coldtank(ux = {258.45, 280.65, 301.05}, nx1 = 3, nx2 = 5, nx3 = 3, table = {{{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, -146.5, -143.248, -140.775, -138.517, -136.648}, {140000000, -162.635, -161.035, -157.64, -155.494, -152.035}, {150000000, -180.028, -177.846, -173.701, -170.903, -169.002}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, -141.183, -138.56, -136.369, -133.951, -132.245}, {140000000, -158.131, -155.37, 152.876, -150.628, -147.527}, {150000000, -175.585, -172.055, -169.349, -166.289, -163.654}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, -137.262, -134.846, -132.417, -130.75, -128.724}, {140000000, -154.456, -151.168, -147.828, -145.605, -144.165}, {150000000, -169.74, -166.643, -164.308, -162.434, -159.555}}}) 
        annotation (Placement(transformation(origin = {-72.00000000000003, 35.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Utilities.Interpolations.Interpolation_3D mflow_gasTank(nx3 = 3, ux = {258.45, 280.65, 301.05},
        nx1 = 3, nx2 = 5, table = {{{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 202.815, 198.508, 195.177, 192.018, 189.319}, {140000000, 225.141, 222.876, 218.481, 215.422, 210.837}, {150000000, 249.36, 246.288, 240.78, 236.938, 234.13}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 196.1, 192.554, 189.399, 186.142, 183.925}, {140000000, 219.731, 215.952, 212.477, 209.174, 205.174}, {150000000, 243.762, 239.025, 235.328, 231.114, 227.496}}, {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {130000000, 191.323, 187.964, 184.512, 182.48, 179.658}, {140000000, 215.171, 210.705, 206.277, 203.252, 201.153}, {150000000, 236.635, 232.428, 229.122, 226.428, 222.501}}}) annotation (Placement(transformation(origin = {-72.00000000000003, -1.0000000000000355},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Interfaces.RealInput T_env 
        annotation (Placement(transformation(origin = {-116.00000000000003, 89.99999999999999},
          extent = {{-14.0, -14.0}, {14.000000000000014, 14.0}})));
      Modelica.Blocks.Tables.CombiTable2D mflow_cold(table = {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {50000000, 101.886, 95.813, 95.5328, 96.104, 99.42}, {75000000, 129.247, 133.09, 129.246, 129.604, 129.272}, {100000000, 163.478, 163.481, 163.474, 163.474, 163.474}, {125000000, 196.777, 196.777, 196.776, 196.777, 197.782}, {150000000, 238.641, 232.349, 229.418, 229.418, 229.418}}) 
        annotation (Placement(transformation(origin = {-72.00000000000003, -49.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Tables.CombiTable2D mflow_hot(table = {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {50000000, -101.886, -95.813, -95.5328, -96.104, -99.42}, {75000000, -129.247, -133.09, -129.246, -129.604, -129.272}, {100000000, -163.478, -163.481, -163.474, -163.474, -163.474}, {125000000, -196.777, -196.777, -196.776, -196.777, -197.782}, {150000000, -238.641, -232.349, -229.418, -229.418, -229.418}}) 
        annotation (Placement(transformation(origin = {-72.00000000000003, -79.00000000000004},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      Modelica.Blocks.Tables.CombiTable2D mflow_gas(table = {{0, 7160000, 7870000, 8580000, 9290000, 10000000}, {50000000, -143.077, -134.395, -133.994, -134.811, -139.543}, {75000000, -181.197, -186.67, -181.196, -181.705, -181.232}, {100000000, -229.166, -229.17, -229.16, -229.161, -229.16}, {125000000, -275.83, -275.83, -275.83, -275.83, -275.837}, {150000000, -334.733, -325.748, -321.574, -321.574, -321.574}}) 
        annotation (Placement(transformation(origin = {-72.00000000000003, -109.00000000000009},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, 0.0},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-100.0, 100.0}, {100.0, -100.0}}), Polygon(origin = {52.00000000000001, 45.99999999999999},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        lineThickness = 1.0,
        points = {{-14.0, 14.0}, {-14.0, -12.0}, {14.0, -28.0}, {14.0, 28.0}}), Polygon(origin = {-50.00000000000001, 45.99999999999999},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        lineThickness = 1.0,
        points = {{14.0, 14.0}, {14.0, -12.0}, {-14.0, -28.0}, {-14.0, 28.0}}), Line(origin = {1.000000000000007, 45.999999999999986},
        points = {{-37.0, 0.0}, {37.0, 0.0}},
        thickness = 0.5), Polygon(origin = {1.0000000000000089, 46.0},
        fillColor = {215, 215, 215},
        fillPattern = FillPattern.HorizontalCylinder,
        points = {{-17.0, 2.0}, {17.0, 2.0}, {17.0, -2.0}, {-17.0, -2.0}}), Rectangle(origin = {1.000000000000007, -8.0},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        lineThickness = 1.0,
        extent = {{-37.0, 16.0}, {36.99999999999999, -16.0}}), Line(origin = {0.9999999999999964, -9.0},
        points = {{-37.0, 1.0}, {-17.0, 1.0}, {-9.0, -9.0}, {7.0, 9.0}, {17.0, -1.0}, {37.0, -1.0}},
        thickness = 1.0), Line(origin = {-36.0, 21.0},
        points = {{0.0, 13.0}, {0.0, -13.0}},
        color = {0, 0, 0},
        thickness = 1.0,
        arrow = {Arrow.None, Arrow.Filled}), Line(origin = {38.00000000000001, 21.0},
        points = {{-7.10543e-15, -13.0}, {0.0, 13.0}},
        color = {0, 0, 0},
        thickness = 1.0,
        arrow = {Arrow.None, Arrow.Filled}), Line(origin = {-64.0, 5.0},
        points = {{-7.10543e-15, -13.0}, {0.0, 13.0}},
        color = {0, 0, 0},
        thickness = 1.0,
        arrow = {Arrow.None, Arrow.Filled}), Line(origin = {66.00000000000001, 79.99999999999997},
        points = {{-1.4210854715202004e-14, -8.0}, {0.0, 8.0}},
        color = {0, 0, 128},
        thickness = 1.0,
        arrow = {Arrow.None, Arrow.Filled}), Text(origin = {-56.00000000000001, -3.9999999999999964},
        lineColor = {0, 0, 0},
        extent = {{-6.0, 3.0}, {6.0, -3.0}},
        textString = "Air In",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None), Text(origin = {75.0, 90.0},
        lineColor = {0, 0, 0},
        extent = {{-6.999999999999986, 4.499999999999998}, {6.999999999999993, -4.499999999999998}},
        textString = "Air Out",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None), Ellipse(origin = {1.0000000000000107, -72.0},
        fillColor = {255, 255, 255},
        lineThickness = 0.5,
        extent = {{-43.0, 24.0}, {43.0, -24.0}}), Ellipse(origin = {88.0, 47.99999999999999},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-10.0, 10.0}, {10.0, -10.0}}), Line(origin = {-70.0, 46.0},
        points = {{-6.0, 0.0}, {6.0, 0.0}},
        thickness = 0.5), Line(origin = {72.0, 48.0},
        points = {{-6.0, 0.0}, {6.0, 0.0}},
        thickness = 0.5), Ellipse(origin = {-86.0, 45.99999999999999},
        fillColor = {255, 255, 255},
        fillPattern = FillPattern.Solid,
        extent = {{-10.0, 10.0}, {10.0, -10.0}}), Text(origin = {88.0, 48.0},
        lineColor = {0, 0, 0},
        extent = {{-6.0, 3.0}, {6.0, -3.0}},
        textString = "ALT",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None), Text(origin = {-86.0, 45.999999999999986},
        lineColor = {0, 0, 0},
        extent = {{-10.0, 4.500000000000007}, {9.999999999999986, -4.499999999999993}},
        textString = "M",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None), Line(origin = {0.0, -36.0},
        points = {{0.0, -10.0}, {0.0, 10.0}},
        color = {0, 0, 0},
        thickness = 1.0,
        arrow = {Arrow.Filled, Arrow.Filled}), Text(origin = {26.0, -17.0},
        lineColor = {0, 0, 0},
        extent = {{-6.0, 3.0}, {6.0, -3.0}},
        textString = "HEX",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None), Text(origin = {1.000000000000007, -70.0},
        lineColor = {0, 0, 0},
        extent = {{-14.0, 10.0}, {14.0, -10.0}},
        textString = "Curve",
        textStyle = {TextStyle.None},
        textColor = {0, 0, 0},
        horizontalAlignment = LinePattern.None)}),
        Diagram(coordinateSystem(extent = {{-160.0, -140.0}, {180.0, 134.0}},
          grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, -3.0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid,
          lineThickness = 2.0,
          extent = {{100.0, 137.0}, {-100.0, -137.0}}), Rectangle(origin = {-66.00000000000004, 54.0},
          lineColor = {255, 0, 0},
          fillColor = {255, 255, 255},
          pattern = LinePattern.Dash,
          fillPattern = FillPattern.Solid,
          lineThickness = 2.0,
          extent = {{-28.000000000000007, 73.0}, {28.000000000000007, -73.0}}), Rectangle(origin = {-66.00000000000007, -79.00000000000001},
          lineColor = {255, 0, 0},
          fillColor = {255, 255, 255},
          pattern = LinePattern.Dash,
          fillPattern = FillPattern.Solid,
          lineThickness = 2.0,
          extent = {{-28.000000000000007, 52.0}, {28.000000000000007, -52.0}})}));



      BoundaryConditions.HeatBoundary.BoundaryTemperature boundaryTemperature(use_T_in = true) 
        annotation (Placement(transformation(origin = {11.999999999999993, -40.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    equation
      if u_dispatch > 0 then
        mflow_hottank.x1 = u_dispatch * P_cap;
        T_hottank.x1 = u_dispatch * P_cap;
        mflow_coldtank.x1 = u_dispatch * P_cap;
        mflow_gasTank.x1 = u_dispatch * P_cap;
        mflow_cold.u1 = 0;
        mflow_hot.u1 = 0;
        mflow_gas.u1 = 0;

        mflow_hottank.y = Mdot_h1.mflow_in;
        T_hottank.y = Mdot_h1.T_in;
        mflow_coldtank.y = Mdot_c2.mflow_in;
        mflow_gasTank.y = Mdot_g1.mflow_in;
        0 = Mdot_c1.mflow_in;
        0 = Mdot_h2.mflow_in;
        0 = Mdot_g2.mflow_in;

      elseif u_dispatch < 0 then
        mflow_hottank.x1 = 0;
        T_hottank.x1 = 0;
        mflow_coldtank.x1 = 0;
        mflow_gasTank.x1 = 0;
        mflow_cold.u1 = -u_dispatch * P_cap;
        mflow_hot.u1 = -u_dispatch * P_cap;
        mflow_gas.u1 = -u_dispatch * P_cap;

        0 = Mdot_h1.mflow_in;
        T_hottank.y = Mdot_h1.T_in;
        0 = Mdot_c2.mflow_in;
        0 = Mdot_g1.mflow_in;
        mflow_cold.y = Mdot_c1.mflow_in;
        mflow_hot.y = Mdot_h2.mflow_in;
        mflow_gas.y = Mdot_g2.mflow_in;
        // connect(mflow_hottank.y, Mdot_h1.mflow_in);
        // connect(T_hottank.y, Mdot_h1.T_in);
        // connect(mflow_coldtank.y, Mdot_c2.mflow_in);
        // connect(mflow_gasTank.y, Mdot_g1.mflow_in);
        // connect(mflow_cold.y, Mdot_c1.mflow_in);
        // connect(mflow_hot.y, Mdot_h2.mflow_in);
        // connect(mflow_gas.y, Mdot_g2.mflow_in);
      else
        mflow_hottank.x1 = 0;
        T_hottank.x1 = 0;
        // T_hottank.x1 = 500;
        mflow_coldtank.x1 = 0;
        mflow_gasTank.x1 = 0;
        mflow_cold.u1 = 0;
        mflow_hot.u1 = 0;
        mflow_gas.u1 = 0;

        Mdot_h1.mflow_in = 0;
        Mdot_h1.T_in = T_hottank.y;
        Mdot_c2.mflow_in = 0;
        Mdot_g1.mflow_in = 0;
        Mdot_c1.mflow_in = 0;
        Mdot_h2.mflow_in = 0;
        Mdot_g2.mflow_in = 0;
      end if;
      PBS.P_plan = u_dispatch * P_cap;
      if PBS.P_act >= 0 then
        PBS.C = -PBS.P_act * PBS.c1 / 3.6e6 + C_GasTank_penality + C_HotTank_penality + C_ColdTank_penality;
        C_GasTank_penality = k * E_cap * (e ^ (-min(gastank.SOC - gastank.SOC_min, 0)) - 1) + k * (e ^ (-min(gastank.SOC_max - gastank.SOC, 0)) - 1);
        C_HotTank_penality = k * E_cap * (e ^ (-min(hottank.SOC - hottank.SOC_min, 0)) - 1) + k * (e ^ (-min(hottank.SOC_max - hottank.SOC, 0)) - 1);
        C_ColdTank_penality = k * E_cap * (e ^ (-min(coldtank.SOC - coldtank.SOC_min, 0)) - 1) + k * (e ^ (-min(coldtank.SOC_max - coldtank.SOC, 0)) - 1);
      else
        PBS.C = -PBS.P_act * PBS.c2 / 3.6e6 + C_GasTank_penality + C_HotTank_penality + C_ColdTank_penality;
        C_GasTank_penality = k * E_cap * (e ^ (-min(gastank.SOC - gastank.SOC_min, 0)) - 1) + k * (e ^ (-min(gastank.SOC_max - gastank.SOC, 0)) - 1);
        C_HotTank_penality = k * E_cap * (e ^ (-min(hottank.SOC - hottank.SOC_min, 0)) - 1) + k * (e ^ (-min(hottank.SOC_max - hottank.SOC, 0)) - 1);
        C_ColdTank_penality = k * E_cap * (e ^ (-min(coldtank.SOC - coldtank.SOC_min, 0)) - 1) + k * (e ^ (-min(coldtank.SOC_max - coldtank.SOC, 0)) - 1);
      end if;
      PBS.P_act = PBS.P_plan;

      PBS.Capex = Capex;
      PBS.c2 = c_sale;
      PBS.c1 = c_buy;

      der(Income) = PBS.C;
      gastank.p = mflow_hottank.x2;
      gastank.p = T_hottank.x2;
      gastank.p = mflow_coldtank.x2;
      gastank.p = mflow_gasTank.x2;
      gastank.p = mflow_cold.u2;
      gastank.p = mflow_hot.u2;
      gastank.p = mflow_gas.u2;
      boundaryTemperature.T_in = T_env;
      connect(Mdot_h1.fluidPort, hottank.port_a) 
        annotation (Line(origin = {11.999999999999964, 67.99999999999999},
          points = {{-13.000000000000007, 1.4210854715202004e-14}, {13.000000000000032, 1.4210854715202004e-14}},
          color = {0, 127, 255}));
      connect(hottank.port_b, Mdot_h2.fluidPort) 
        annotation (Line(origin = {55.99999999999997, 67.99999999999999},
          points = {{-12.999999999999972, 1.4210854715202004e-14}, {13.000000000000028, 1.4210854715202004e-14}},
          color = {0, 127, 255}));
      connect(Mdot_c2.fluidPort, coldtank.port_b) 
        annotation (Line(origin = {55.99999999999997, 22.999999999999986},
          points = {{13.000000000000028, -1.4210854715202004e-14}, {-12.999999999999972, 7.105427357601002e-15}},
          color = {0, 127, 255}));
      connect(coldtank.port_a, Mdot_c1.fluidPort) 
        annotation (Line(origin = {11.999999999999964, 22.999999999999986},
          points = {{13.000000000000032, 7.105427357601002e-15}, {-13.000000000000007, 0.0}},
          color = {0, 127, 255}));
      connect(Mdot_g1.fluidPort, gastank.port_a) 
        annotation (Line(origin = {12.000000000000078, -68.0},
          points = {{-13.000000000000007, 0.0}, {12.000000000000021, 0.0}},
          color = {0, 127, 255}));
      connect(gastank.port_b, Mdot_g2.fluidPort) 
        annotation (Line(origin = {58.00000000000007, -68.0},
          points = {{-14.199999999999967, 0.0}, {14.999999999999986, 0.0}},
          color = {0, 127, 255}));
      // connect(mflow_hottank.y, Mdot_h1.mflow_in)
      //   annotation (Line(origin = {2.0, 69.0}, 
      //     points = {{-48.99999999999999, 0.9999999999999858}, {36.0, 0.9999999999999858}, {36.0, -6.999999999999993}}, 
      //     color = {0, 0, 127}));
      // connect(T_hottank.y, Mdot_h1.T_in)
      //   annotation (Line(origin = {8.0, 51.0}, 
      //     points = {{-54.99999999999999, 54.99999999999997}, {42.0, 54.99999999999997}, {42.0, 11.000000000000007}}, 
      //     color = {0, 0, 127}));
      // connect(mflow_coldtank.y, Mdot_c2.mflow_in)
      //   annotation (Line(origin = {2.0, 10.0}, 
      //     points = {{-48.99999999999999, 23.999999999999993}, {136.00000000000006, 23.999999999999993}, {136.00000000000006, 6.999999999999986}}, 
      //     color = {0, 0, 127}));
      // connect(mflow_gasTank.y, Mdot_g1.mflow_in)
      //   annotation (Line(origin = {2.0, -30.0}, 
      //     points = {{-48.99999999999999, 27.99999999999997}, {-32.0, 27.99999999999997}, {-32.0, 12.0}, {36.00000000000001, 12.0}, {36.00000000000001, 2.0}}, 
      //     color = {0, 0, 127}));
      // connect(mflow_cold.y, Mdot_c1.mflow_in)
      //   annotation (Line(origin = {-3.999999999999993, -2.999999999999986}, 
      //     points = {{-42.99999999999999, -47.0}, {4.0, -47.0}, {4.0, 19.99999999999998}, {41.99999999999999, 19.99999999999998}}, 
      //     color = {0, 0, 127}));
      // connect(mflow_hot.y, Mdot_h2.mflow_in)
      //   annotation (Line(origin = {46.000000000000014, 5.000000000000014}, 
      //     points = {{-93.0, -85.00000000000004}, {128.0, -85.00000000000004}, {128.0, 101.0}, {92.00000000000004, 101.0}, {92.00000000000004, 56.999999999999986}}, 
      //     color = {0, 0, 127}));
      // connect(mflow_gas.y, Mdot_g2.mflow_in)
      //   annotation (Line(origin = {48.000000000000014, -54.999999999999986}, 
      //     points = {{-95.0, -55.00000000000013}, {111.99999999999999, -55.00000000000013}, {111.99999999999999, 34.999999999999986}, {93.99999999999999, 34.999999999999986}, {93.99999999999999, 26.999999999999986}}, 
      //     color = {0, 0, 127}));
      connect(T_env, T_hottank.x3);
      connect(mflow_hottank.x3, T_env);
      connect(mflow_coldtank.x3, T_env);
      connect(mflow_gasTank.x3, T_env);
      connect(boundaryTemperature.port[1], gastank.heatport) 
        annotation (Line(origin = {303.00000000000017, -117.00000000000003},
          points = {{-281.00000000000017, 77.00000000000003}, {-269.06492753623195, 77.00000000000003}, {-269.06492753623195, 58.800000000000026}},
          color = {191, 0, 0}));
    end CompressedAirEnergyStorage;
    model Tank "水箱模型"
      // 介质
      replaceable package Medium = TypicalScensrio.Media.FluidMedia.Water 
         constrainedby TypicalScensrio.Media.FluidMedia.PartialMedium "介质" 
          annotation (choicesAllMatching = true);

      // 结构参数
      parameter Modelica.SIunits.Area A = 100 "横截面积";
      parameter Modelica.SIunits.Volume V0 = 5000 "总容积";
      parameter Real SOC_max = 0.95 "容积上限";
      parameter Real SOC_min = 0.05 "容积下限";


      // 初始化
      parameter Modelica.SIunits.Temperature T_start = 473.15 "初始温度" 
        annotation (Dialog(tab = "初始化"));
      parameter Modelica.SIunits.SpecificEnthalpy h_start = Medium.h_pT(p, T_start) "初始比焓" 
        annotation (Dialog(tab = "初始化"));
      parameter Modelica.SIunits.Length level_start = 10 "初始水位" 
        annotation (Dialog(tab = "初始化"));

      Real SOC;
      Modelica.SIunits.Length level(start = max(level_start, eps), stateSelect = StateSelect.prefer) "液位";
      Modelica.SIunits.Volume V "液体容积";
      Modelica.SIunits.Mass m(start = level_start * A * 1000, fixed = true) "液体质量";
      Modelica.SIunits.Temperature T "液体温度";
      Modelica.SIunits.InternalEnergy U "液体内能";
      Modelica.SIunits.SpecificEnthalpy h(start = h_start, stateSelect = StateSelect.prefer) "液体比焓";
      parameter Modelica.SIunits.Pressure p = 16e5 "容器压力";

      // Real SOC(start = 0.5) "容积比";
      TypicalScensrio.Interfaces.Fluid.FluidPort_a port_a(
        redeclare package Medium = Medium) "接口A" annotation (Placement(transformation(origin = {-60.0, -108.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}}),
          iconTransformation(origin = {-90.00000000000003, 1.4210854715202004e-14},
            extent = {{-10.0, -10.0}, {10.0, 10.0}})));

      TypicalScensrio.Interfaces.Fluid.FluidPort_b port_b(
        redeclare package Medium = Medium) "接口B" annotation (Placement(transformation(origin = {58.0, -108.0},
          extent = {{-20.0, -20.0}, {20.0, 20.0}}),
          iconTransformation(origin = {90.0, 1.4210854715202004e-14},
            extent = {{-10.0, -10.0}, {10.0, 10.0}})));

      parameter Boolean use_HeatPort = false "true-使用换热" annotation (Dialog(tab = "高级", group = "换热"), Evaluate = true);
      parameter Modelica.SIunits.HeatFlowRate Q_gen = 0 "内热能" annotation (Dialog(tab = "高级", group = "换热"));
    protected
      Modelica.SIunits.Density rho = Medium.rho_ph(p, h);

      annotation (Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        grid = {2.0, 2.0})),
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Rectangle(origin = {0.0, 0.0},
          lineColor = {0, 0, 0},
          fillColor = {0, 0, 0},
          fillPattern = FillPattern.Solid,
          extent = {{-80.0, 80.0}, {80.0, -80.0}}), Rectangle(origin = {0.0, 0.0},
          lineColor = {0, 0, 0},
          fillColor = {255, 255, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-78.0, 78.0}, {78.0, -78.0}}), Rectangle(origin = {2.842170943040401e-14, -7.0},
          lineColor = {0, 0, 0},
          fillColor = {0, 85, 255},
          fillPattern = FillPattern.Solid,
          extent = DynamicSelect({{-78, -71.0}, {78, 71}}, {{-78, -71.0}, {78, (-71 + level / V0 * A * 156)}})), Text(origin = {-3.552713678800501e-15, 115.99999999999997},
          lineColor = {0, 0, 0},
          extent = {{-86.0, 32.00000000000003}, {86.0, -31.99999999999997}},
          textString = "%name",
          textStyle = {TextStyle.Bold},
          textColor = {0, 0, 0},
          horizontalAlignment = LinePattern.None)}),
        experiment(Algorithm = Euler, IntegratorStep = 72, Interval = 3600, StartTime = 0, StopTime = 3.1536e+07, Tolerance = 0.0001));
    equation
      // assert(level <= SOC_max * V0 / A, "储罐容量超过上限");
      // assert(level > SOC_min * V0 / A, "储罐容量超过下限");
      V = A * level;
      m = V * 1000;
      U = m * Medium.u_ph(p, h);
      SOC = level / (V0 / A);
      if noEvent(SOC > SOC_min and SOC < SOC_max) then
        der(m) = port_a.m_flow + port_b.m_flow;
        // V * (p / rho * Medium.drhodh_p_ph(p, h) + rho) * der(h) = port_a.m_flow * (inStream(port_a.h_outflow) - (h - p / rho)) + port_b.m_flow * (port_b.h_outflow - (h - p / rho)) + Q_gen;
        m * der(h) = port_a.m_flow * inStream(port_a.h_outflow) + port_b.m_flow * port_b.h_outflow + Q_gen;
      else
        der(m) = 0;
        der(h) = 0;
      end if;
      T = Medium.T_ph(p, h);
      // SOC = level / (V0 / A);
      // 组分平衡
      port_a.h_outflow = inStream(port_a.h_outflow);
      port_b.h_outflow = h;
      port_a.p = p;
      port_b.p = p;

      port_a.Xi_outflow = Medium.reference_X;
      port_b.Xi_outflow = Medium.reference_X;
      inStream(port_a.Exergy) = port_b.Exergy;
      port_a.Exergy = inStream(port_b.Exergy);
      inStream(port_a.C_exergy) = port_b.C_exergy;
      port_a.C_exergy = inStream(port_b.C_exergy);
      port_a.Xk = inStream(port_a.Xk);
      port_b.Xk = inStream(port_a.Xk);
      annotation (Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
        preserveAspectRatio = false,
        grid = {2.0, 2.0}), graphics = {Text(origin = {-1.0, 114.0},
        lineColor = {0, 0, 255},
        extent = {{-150.0, 20.0}, {150.0, -20.0}},
        textString = "%name",
        textColor = {0, 0, 255})}),
        Diagram(coordinateSystem(
          preserveAspectRatio = true, extent = {{-100, -100}, {100, 100}})));
    end Tank;
    model GasTank "储气罐"
      replaceable package Medium = TypicalScensrio.Media.FluidMedia.IdealAir 
         constrainedby TypicalScensrio.Media.FluidMedia.PartialMedium 
          annotation (Dialog(tab = "参数设置", group = "工质选择"), choicesAllMatching = true);
      parameter SI.Pressure p_norm = 100e5 "额定储气压力";
      // parameter SI.Pressure p_min = 10e5 "压力下限" ;
      parameter Real SOC_max = 1 "SOC上限";
      parameter Real SOC_min = 0.6 "SOC下限";
      parameter SI.Volume V = 426618.1513 "气瓶容积" annotation (Dialog(tab = "参数设置", group = "结构参数"));
      parameter SI.Length d = 1.4 "直径" annotation (Dialog(tab = "参数设置", group = "结构参数"));
      parameter SI.CoefficientOfHeatTransfer alpha_w = 4 "气体与壁面的传热系数" annotation (Dialog(tab = "参数设置", group = "性能参数"));
      parameter SI.Pressure p_start = 7.115999999999999e6 "初始压力" 
        annotation (Dialog(tab = "初始条件"));
      parameter SI.Temperature T_start = 280.65 "初始温度" 
        annotation (Dialog(tab = "初始条件"));
      parameter SI.SpecificEnthalpy h_start = Medium.h_pT(p_start, T_start) annotation (Dialog(tab = "初始条件"));
      final parameter SI.Area S = 4 * V / d "气瓶表面积";

      final parameter SI.SpecificHeatCapacity R = Medium.R() "气体常数";
      final parameter SI.Mass m_g_start = Medium.rho_pT(p_start, T_start) * V;
      SI.Density rho "气体密度";
      SI.SpecificEnthalpy h(start = h_start, stateSelect = StateSelect.always) "气体比焓";
      SI.HeatFlowRate q_w "气瓶与壁面的传热";
      SI.Temperature T_w(start = T_start) "壁面温度";
      SI.Temperature T(start = T_start) "气体温度";
      SI.Mass m_g(start = m_g_start) "气体质量";
      SI.Pressure p(start = p_start, fixed = true, stateSelect = StateSelect.always) "气体压力";
      Real SOC;

      TypicalScensrio.Interfaces.Fluid.FluidPort_a port_a(redeclare package Medium = Medium) annotation (Placement(transformation(origin = {-100.0, -6.661338147750939e-16},
        extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      TypicalScensrio.Interfaces.Thermal.HeatPort_a heatport 
        annotation (Placement(transformation(origin = {-0.6492753623188467, 98.0},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
      import SI = Modelica.SIunits;
      Interfaces.Fluid.FluidPort_b port_b(
        redeclare package Medium = Medium) annotation (Placement(transformation(origin = {98.00000000000001, -3.3306690738754696e-16},
          extent = {{-10.0, -10.0}, {10.0, 10.0}})));
    equation
      assert(SOC <= SOC_max, "气库容量超过上限");
      assert(SOC > SOC_min, "气库容量超过下限");

      SOC = p / p_norm;
      //质量守恒方程
      V * (Medium.drhodp_h_ph(p, h) * der(p) + Medium.drhodh_p_ph(p, h) * der(h)) = port_a.m_flow + port_b.m_flow;
      //计算密度
      port_a.p = p;
      port_b.p = p;
      rho = m_g / V;
      rho = Medium.rho_ph(p, h);

      //能量方程
      q_w = alpha_w * S * (T_w - T);
      V * (-der(p) + Medium.drhodp_h_ph(p, h) * der(p) * h + Medium.drhodh_p_ph(p, h) * der(h) * h + rho * der(h)) = port_a.m_flow * inStream(port_a.h_outflow) + port_b.m_flow * port_b.h_outflow + q_w;

      //比焓方程
      port_a.h_outflow = h;
      port_b.h_outflow = h;
      T = max(Medium.T_ph(p, h), 253);
      // T = 300;
      //壁面温度
      heatport.T = T_w;
      q_w = heatport.Q_flow;

      //气体状态方程
      port_a.Xi_outflow = Medium.reference_X;
      port_b.Xi_outflow = Medium.reference_X;
      inStream(port_a.Exergy) = port_b.Exergy;
      port_a.Exergy = inStream(port_b.Exergy);
      inStream(port_a.C_exergy) = port_b.C_exergy;
      port_a.C_exergy = inStream(port_b.C_exergy);
      port_a.Xk = inStream(port_a.Xk);
      port_b.Xk = inStream(port_a.Xk);
      annotation (
        Icon(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          grid = {2.0, 2.0}), graphics = {Ellipse(origin = {-0.6492753623188463, 0.32463768115940184},
          lineColor = {40, 40, 40},
          fillColor = {100, 255, 255},
          fillPattern = FillPattern.Solid,
          extent = {{-98.82898550724639, 98.504347826087}, {98.82898550724637, -98.50434782608694}},
          thickness = 0.25)}),
        Diagram(coordinateSystem(extent = {{-100.0, -100.0}, {100.0, 100.0}},
          preserveAspectRatio = false,
          grid = {2.0, 2.0})));
    end GasTank;
    model DispatchOrder_HD "调度指令"
      parameter Integer N_Time = 8 "单次优化时间";
      parameter Real DischargeUL = 1 "放电最大功率";
      parameter Real DischargeBL = 1 / 3 "放电最小功率";

      parameter Real PowerInput[N_Time] = {0.5, -0.2, 1, -0.5, -1, 0.1, -0.9, 0.9};

      Real Power[N_Time];
    equation
      for i in 1:N_Time loop
        Power[i] = (PowerInput[i] - 1) / 2 * (DischargeUL - DischargeBL) - DischargeBL;
      end for;
    end DispatchOrder_HD;
    model Model1
      annotation(__MWORKS(version="2025b"),Diagram(coordinateSystem(extent={{-100,-100},{100,100}},
grid={2,2})));
      GasTank gastank(redeclare package Medium = Media.FluidMedia.IdealAir

        ) 
        annotation (Placement(transformation(origin={-36,-55},
extent={{-10,-10},{10,10}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_g1(use_mflow_in = false,
        redeclare package Medium = Media.FluidMedia.IdealAir,m_flow=20) annotation (Placement(transformation(origin={-80,-55},
extent={{-10,-10},{10,10}})));
      BoundaryConditions.FluidBoundary.BoundaryMdot Mdot_g2(m_flow = 0,
        use_mflow_in = false, redeclare package Medium = Media.FluidMedia.IdealAir) annotation (Placement(transformation(origin={12,-55},
extent={{10,-10},{-10,10}})));
      BoundaryConditions.HeatBoundary.BoundaryTemperature boundaryTemperature(use_T_in = false) 
        annotation (Placement(transformation(origin={-58,-27},
extent={{-10,-10},{10,10}})));
    equation
      connect(Mdot_g1.fluidPort, gastank.port_a) 
      annotation(Line(origin={-58,-55},
      points={{-13,0},{12,0}},
      color={0,127,255}));
      connect(gastank.port_b, Mdot_g2.fluidPort) 
      annotation(Line(origin={-12,-55},
      points={{-14.2,0},{15,0}},
      color={0,127,255}));
      connect(boundaryTemperature.port[1], gastank.heatport) 
      annotation(Line(origin={233,-104},
      points={{-281,77},{-269.065,77},{-269.065,58.8}},
      color={191,0,0}));

    end Model1;
  end CompressedAirEnergyStorage;
  annotation (Protection(access = Access.icon));
end TypicalScenarios;
