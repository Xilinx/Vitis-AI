..
   Copyright 2019 Xilinx, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: Vitis Quantitative Finance Library, Black-Scholes-Merton, Heston, European, American, Asian, Barrier, Digital, Cliquet, Binomial Tree, Cox-Ross-Rubinstein, Hull-White, Black-Scholes, Monte Carlo
   :description: Vitis quantitative finance library provides pricing engines to calculate price. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


***********************
Pricing Engine Overview
***********************

Vitis Quantitative Finance Library 1.0 provides 12 pricing engines to calculate price for the following options.

* European Option
* American Option
* Asian Option
* Barrier Option
* Digital Option
* Cliquet Option

Additionally, the following options have 2 Closed-Form solution engines; the Black-Scholes-Merton model and the Heston model.

* European Option

There is also a Binomial Tree (Cox-Ross-Rubinstein) engine that will calculate prices for:

* European Option
* American Option


The main feature for each pricing engines is as the following table.

+-------------------------+--------------------+----------------------------+--------------------------+
|Pricing Engines          |Option              |Model                       |Solution Method           |
+-------------------------+--------------------+----------------------------+--------------------------+
|MCEuropeanEngine         |European            |Black-Scholes               | Monte Carlo              |
+-------------------------+--------------------+                            +                          +
|MCAsianAPEngine          |Asian               |                            |                          |
+-------------------------+                    +                            +                          +
|MCAsianGPEngine          |                    |                            |                          |
+-------------------------+                    +                            +                          +
|MCAsianASEngine          |                    |                            |                          |
+-------------------------+--------------------+                            +                          +
|MCCliquetEngine          |Cliquet             |                            |                          |
+-------------------------+--------------------+                            +                          +
|MCDigitalEngine          |Digital             |                            |                          |
+-------------------------+--------------------+                            +                          +
|MCBarrierEngine          |Barrier             |                            |                          |
+-------------------------+                    +                            +                          +
|MCBarrierNoBiasEngine    |                    |                            |                          |
+-------------------------+--------------------+                            +                          +
|MCAmericanEngine         |American            |                            |                          |
+-------------------------+--------------------+----------------------------+                          +
|MCEuropeanHestonEngine   |European            |Heston                      |                          |
+-------------------------+                    +                            +                          +
|MCMultiAssetEuropean/    |                    |                            |                          |
|HestonEngine             |                    |                            |                          |
+-------------------------+--------------------+----------------------------+--------------------------+
|CFBlackScholes           |European            |Black-Scholes               | Closed Form              |
+-------------------------+--------------------+----------------------------+                          +
|CFBlack76                |European            |Black 76                    |                          |
+-------------------------+--------------------+----------------------------+                          +
|CFHeston                 |European            |Heston                      |                          |
+-------------------------+--------------------+----------------------------+--------------------------+
|BTCRR                    |European            |Cox-Ross-Rubinstein         | Binomial Tree            |
|                         |American            |                            |                          |
+-------------------------+--------------------+----------------------------+--------------------------+
|FdHullWhiteEngine        |Swaption            |Hull-White                  |finite-difference methods |
+-------------------------+                    +----------------------------+                          +
|FdG2SwaptionEngine       |                    |Two-additive factor Gaussian|                          |
+-------------------------+                    +----------------+-----------+--------------------------+
|treeSwaptionEngine       |                    |Hull-White                  |Trinomial Tree            |
|                         |                    |Black-Barasinski            |                          |
|                         |                    |Cox-Ingersoll-Ross          |                          |
|                         |                    |Extended Cox-Ingersoll-Ross |                          |
|                         |                    |Vasicek                     |                          |
|                         |                    |Two-additive factor Gaussian|                          |
+-------------------------+--------------------+----------------------------+                          +
|treeSwapEngine           |Swap                |Hull-White                  |Trinomial Tree            |
|                         |                    |Black-Barasinski            |                          |
|                         |                    |Cox-Ingersoll-Ross          |                          |
|                         |                    |Extended Cox-Ingersoll-Ross |                          |
|                         |                    |Vasicek                     |                          |
|                         |                    |Two-additive factor Gaussian|                          |
+-------------------------+--------------------+----------------------------+                          +
|treeCapFloorEngine       |Cap/Floor           |Hull-White                  |Trinomial Tree            |
|                         |                    |Black-Barasinski            |                          |
|                         |                    |Cox-Ingersoll-Ross          |                          |
|                         |                    |Extended Cox-Ingersoll-Ross |                          |
|                         |                    |Vasicek                     |                          |
|                         |                    |Two-additive factor Gaussian|                          |
+-------------------------+--------------------+----------------------------+                          +
|treeCallableBondEngine   |Callable Bond       |Hull-White                  |Trinomial Tree            |
|                         |                    |Black-Barasinski            |                          |
|                         |                    |Cox-Ingersoll-Ross          |                          |
|                         |                    |Extended Cox-Ingersoll-Ross |                          |
|                         |                    |Vasicek                     |                          |
|                         |                    |Two-additive factor Gaussian|                          |
+-------------------------+--------------------+----------------------------+--------------------------+
|MCHullWhiteCapFloorEngine|Cap/Floor           |Hull-White                  |Monte Carlo               |
+-------------------------+--------------------+----------------------------+--------------------------+
|CPICapFloorEngine        |CPI Cap/Floor       | --                         |Close Form                |
+-------------------------+--------------------+----------------------------+                          +
|DiscountingBondEngine    |Discounting Bond    | --                         |                          |
+-------------------------+--------------------+----------------------------+                          +
|InflationCapFloorEngine  |Inflation Cap/Floor | --                         |                          |
+-------------------------+--------------------+----------------------------+--------------------------+
|hjmEngine                | N/A                | Heath-Jarrow-Morton        | Monte Carlo              |
+-------------------------+--------------------+----------------------------+--------------------------+
|lmmEngine                | N/A                | LIBOR Market Model (BGM)   | Monte Carlo              |
+-------------------------+--------------------+----------------------------+--------------------------+
|HWAEngine                |Bond Price          | Hull-White Analytic        | Closed Form              |
|                         |Option              |                            |                          |
|                         |Cap/Floor           |                            |                          |
+-------------------------+--------------------+----------------------------+--------------------------+
