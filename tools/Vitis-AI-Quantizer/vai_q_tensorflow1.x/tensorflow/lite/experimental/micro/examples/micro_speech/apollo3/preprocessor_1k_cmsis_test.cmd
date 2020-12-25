# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Needs to be compiled with -O0
file ../../../tools/make/gen/apollo3evb_cortex-m4/bin/preprocessor_1k_cmsis_test
target remote localhost:2331
load ../../../tools/make/gen/apollo3evb_cortex-m4/bin/preprocessor_1k_cmsis_test
monitor reset
break preprocessor.cc:68
commands
dump verilog value cmsis_windowed_input.txt bufB
c
end
break preprocessor.cc:76
commands
dump verilog value cmsis_dft.txt bufA
c
end
break preprocessor.cc:81
commands
dump verilog value cmsis_power.txt bufB
c
end
break preprocessor.cc:83
commands
dump verilog memory cmsis_power_avg.txt output output+42
c
end
break preprocessor_1k.cc:50
commands
print count
end
c
