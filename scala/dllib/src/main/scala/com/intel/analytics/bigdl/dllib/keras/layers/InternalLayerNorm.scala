/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dllib.keras.layers.internal

import com.intel.analytics.bigdl.dllib.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.TensorOperation

import scala.reflect.ClassTag

private[bigdl] class InternalLayerNorm[T: ClassTag](
   val nOutput: Int = 768, val eps: Double = 1e-5)
 (implicit ev: TensorNumeric[T]) extends TensorModule[T]{
  val weight = Tensor.ones[T](nOutput).view(1, nOutput).rand()
  val bias = Tensor[T](nOutput).view(1, nOutput).rand()

  var gradWeight: Tensor[T] = Tensor[T]()
  var gradBias: Tensor[T] = Tensor[T]()

  var y: Tensor[T] = null
  var divInput1: Tensor[T] = null
  var divInput2: Tensor[T] = null
  var sqrtInput: Tensor[T] = null

  private def optimzedOperation(input1: Tensor[T], input2: Tensor[T], operation: String) = {
    val dim = input1.dim()
    val kk = Array.fill[Int](dim - 1)(1)
    var m = 0
    var cnt = 0

    while (kk(0) < input1.size(1) + 1) {
      cnt += 1
      if (cnt < input1.dim() - 1) {
        m = 1
        while (m < kk.size) {
          kk(m) = 1
          m += 1
        }
        while (kk(1) < input1.size(2) + 1) {
          cnt += 1
          if (cnt < input1.dim() - 1) {
            m = 2
            while (m < kk.size) {
              kk(m) = 1
              m += 1
            }
            while (kk(2) < input1.size(3) + 1) {
              cnt += 1
              if (cnt < input1.dim() - 1) {}
              else {
                if (operation == "-") {
                  input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).narrow(3, kk(2), 1)
                    .sub(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else if (operation == "/") {
                  input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).narrow(3, kk(2), 1)
                    .div(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else if (operation == "+") {
                  input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).narrow(3, kk(2), 1)
                    .add(input2.valueAt(kk(0), kk(1), kk(2), 1))
                } else {
                  input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).narrow(3, kk(2), 1)
                    .mul(input2.valueAt(kk(0), kk(1), kk(2), 1))
                }
              }
              kk(2) += 1
              cnt = 2
            }
          } else {
            if (operation == "-") {
              input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).sub(input2.valueAt(kk(0), kk(1), 1))
            } else if (operation == "/") {
              input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).div(input2.valueAt(kk(0), kk(1), 1))
            } else if (operation == "+") {
              input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).add(input2.valueAt(kk(0), kk(1), 1))
            } else {
              input1.narrow(1, kk(0), 1).narrow(2, kk(1), 1).mul(input2.valueAt(kk(0), kk(1), 1))
            }
          }
          kk(1) += 1
          cnt = 1
        }
      } else {
        if (operation == "-") {
          input1.narrow(1, kk(0), 1).sub(input2.valueAt(kk(0), 1))
        } else if (operation == "/") {
          input1.narrow(1, kk(0), 1).div(input2.valueAt(kk(0), 1))
        } else if (operation == "+") {
          input1.narrow(1, kk(0), 1).add(input2.valueAt(kk(0), 1))
        } else {
          input1.narrow(1, kk(0), 1).mul(input2.valueAt(kk(0), 1))
        }
      }
      kk(0) += 1
      cnt = 0
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    // First
    val u = input.sum(dim).div(ev.fromType(input.size(dim)))
    val shiftInput = input.clone()
    optimzedOperation(shiftInput, u, "-")
    divInput1 = shiftInput // TensorOperation.subTensor(input.clone(), u)
    val square = divInput1.clone().square()
    val s = square.sum(square.dim()).div(ev.fromType(square.size(square.dim())))
    sqrtInput = s.add(ev.fromType(eps))
    divInput2 = sqrtInput.clone().sqrt()
    val shiftInput2 = divInput1.clone()
    optimzedOperation(shiftInput2, divInput2, "/")
    y = shiftInput2
    output = y.clone().cmul(weight).add(bias)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val divGradInput1 = gradOutput.clone().cmul(weight)
    optimzedOperation(divGradInput1, divInput2, "/")

    val divGradInput2k = divGradInput1.clone()
    optimzedOperation(divGradInput2k, divInput2, "/")
    val divGradInput2 = divGradInput2k.cmul(divInput1)

    val squareGadO = divGradInput2.sum(divGradInput2.dim())
    val divInput2Aux = divInput2.clone()
    optimzedOperation(divInput2Aux, sqrtInput, "/")
    val sqrtGradI = divInput2Aux.cmul(squareGadO.contiguous())

    val sumGradI = sqrtGradI.div(ev.fromType(-1 * divInput1.size(divInput1.dim())))
    // .expand(divInput1.size())

    val divInput1Aux = divInput1.clone()
    optimzedOperation(divInput1Aux, sumGradI, "*")
    val squareGradI = divInput1Aux

    val addGradO = divGradInput1.add(squareGradI)
    val addGradI = addGradO.sum(addGradO.dim())
    val negativeGradO = addGradI.sum(addGradI.dim())
    //    val negativeGradI = negativeGradO.mul(ev.fromType(-1))
    val sum2GradI = negativeGradO.div(ev.fromType(-1 * input.size(input.dim())))
    optimzedOperation(addGradO, sum2GradI, "+")
    gradInput = addGradO
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    var i = 1
    gradWeight = y.clone().cmul(gradOutput)
    gradBias = gradOutput
    while (i < gradOutput.dim()) {
      gradBias = gradBias.sum(i)
      gradWeight = gradWeight.sum(i)
      i += 1
    }
    gradBias.resize(bias.size())
    gradWeight.resize(weight.size())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }
}
