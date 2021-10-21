/*
 * Copyright 2019 Xilinx, Inc.
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

#ifndef XF_DATABASE_DYNAMIC_ALU_HOST_H
#define XF_DATABASE_DYNAMIC_ALU_HOST_H

#include <ap_int.h>
#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <string>

namespace xf {
namespace database {
namespace details {
#define Formula_Max_Length 256
#define Dynamic_ALU_Max_Tree_Height 4

//------------- generate polish notation ---------------------------//

// whether Operator or not
inline bool isOperator(char op) {
    return (op == ':' || op == '?' || op == '+' || op == '*' || op == '=' || op == '!' || op == '>' || op == '<' ||
            op == '|' || op == '&' || op == '^' || op == '~' || op == 'X' || op == 'N' || op == 'O' || op == 'R');
}

// whether Operand or not
inline bool isOperand(char op) {
    return (op == 's' || op == 't' || op == 'r' || op == 'm' || op == 'c' || op == '1' || op == '2' || op == '3' ||
            op == '4' || op == '-');
}

inline bool isCharValid(char op) {
    return (op == '(' || op == ')' || op == ' ');
}

// linked token list
struct Token {
    std::string data = "";
    Token* next = NULL;
};

// clear linked list
inline void clear_Token(Token* head) {
    Token* p;

    while (head) {
        p = head->next;
        delete head;
        head = p;
    }
}

// Remove ' ', differentiate negative sign and subtraction, Tokenize
inline bool Pre_Check(const char* input, Token& Formula) {
    bool check_result = 1;
    int outLen = 0;
    int i = 0;

    char temp1[Formula_Max_Length];
    char temp2[Formula_Max_Length];

    Token *end, *token_temp;
    end = &Formula;

    // scan for not supported operand or operation
    while (input[i] != '\0') {
        if (isOperand(input[i]) || isOperator(input[i]) || isCharValid(input[i])) {
            // true
            check_result &= 1;

            // scan for ' ' and remove
            if (input[i] == ' ') {
                // skip and scan next
            } else {
                // output
                temp1[outLen++] = input[i];
            }
        } else {
            // false
            check_result &= 0;
        }
        i++;
    }
    temp1[outLen] = '\0';

    if (check_result) {
        // scan for '-' to differentiate negative sign and subtraction
        outLen = 0;
        i = 0;
        while (temp1[i] != '\0') {
            if ((isOperand(temp1[i]) && temp1[i + 1] == '-') || (temp1[i] == ')' && temp1[i + 1] == '-')) {
                // insert '+' when '-' stand for subtraction
                temp2[outLen++] = temp1[i];
                temp2[outLen++] = '+';
                temp2[outLen++] = '-';
                i++;
            } else {
                // output
                temp2[outLen++] = temp1[i];
            }
            i++;
        }
        temp2[outLen] = '\0';

        // tokenize input formula in a linked list
        outLen = 0;
        i = 0;
        while (temp2[i] != '\0') {
            // insert a token when '(' or ')'
            if (temp2[i] == '(' || temp2[i] == ')') {
                // new a token
                token_temp = new Token;
                token_temp->data = temp2[i];

                // insert
                end->next = token_temp;
                end = token_temp;
                i++;
            }

            if (isOperator(temp2[i])) {
                // new a token
                token_temp = new Token;

                // scan for multi-byte operator
                while (i < Formula_Max_Length && isOperator(temp2[i])) {
                    token_temp->data += temp2[i];
                    i++;
                }

                // insert
                end->next = token_temp;
                end = token_temp;
            }

            if (isOperand(temp2[i])) {
                // new a token
                token_temp = new Token;

                // scan for multi-byte operand
                while (i < Formula_Max_Length && isOperand(temp2[i])) {
                    token_temp->data += temp2[i];
                    i++;
                }

                // insert
                end->next = token_temp;
                end = token_temp;
            }
        }
    }
    return check_result;
}

// Judge op priority
inline int priority(std::string Operator) {
    int result = 0;

    // mux
    if (Operator == "?") result = 1;
    if (Operator == ":") result = 2;
    if (Operator == "[") result = 2;
    if (Operator == "]") result = 2;

    // boolean algebra
    if (Operator == "||") result = 3;
    if (Operator == "^") result = 4;
    if (Operator == "XOR") result = 4;
    if (Operator == "XNOR") result = 4;
    if (Operator == "&&") result = 5;

    // comparator
    if (Operator == "==") result = 6;
    if (Operator == "!=") result = 6;
    if (Operator == ">") result = 7;
    if (Operator == "<") result = 7;
    if (Operator == ">=") result = 7;
    if (Operator == "<=") result = 7;

    // math
    if (Operator == "+") result = 8;
    if (Operator == "*") result = 9;

    return result;
}

// Generate Reverse Polish Notation
inline void Reverse_Polish(Token& Input_Formula, Token& Output_Formula) {
    // stack is empty when top==0
    std::string stack[Formula_Max_Length]; // step1 initialize a empty stack

    // pointer of stack top
    int top = 0;

    // pointer of input and output formula
    Token *scan, *end, *token_temp;
    scan = Input_Formula.next;
    end = &Output_Formula;

    // processing
    while (scan != NULL) {
        if (isOperand(scan->data.at(0))) // step2 if operand is scanned, put it into output formula
        {
            // new a token
            token_temp = new Token;
            token_temp->data = scan->data;

            // insert
            end->next = token_temp;
            end = end->next;
        }

        if (scan->data == "(") // step3 if '(' is scanned
        {
            top++;
            stack[top] = scan->data;
        }

        while (isOperator(scan->data.at(0))) // step4 if operator is scanned
        {
            if (top == 0 || stack[top] == "(" || priority(scan->data) > priority(stack[top])) {
                // push the operator into stack and break
                top++;
                stack[top] = scan->data;
                break;
            } else {
                // pop stack until top==0 || stack[top]=="(" || priority(scan->data) >
                // priority(stack[top])

                // new a token && pop
                token_temp = new Token;
                token_temp->data = stack[top];
                top--;

                // insert token
                end->next = token_temp;
                end = end->next;
            }
        }

        if (scan->data == ")") // step5 if ')' is scanned, pop stack until '('
        {
            while (stack[top] != "(") {
                // new a token && pop
                token_temp = new Token;
                token_temp->data = stack[top];
                top--;

                // insert token
                end->next = token_temp;
                end = end->next;
            }
            top--;
        }
        scan = scan->next; // step6 return to step2 when formula is not end
    }

    while (top != 0) // step7 pop the residual operator in the stack
    {
        // new a token && pop
        token_temp = new Token;
        token_temp->data = stack[top];
        top--;

        // insert token
        end->next = token_temp;
        end = end->next;
    }
}

//----------generate parse tree && check grammar--------------------------//

// parse tree node
struct Node {
    std::string OP = "";

    bool type = 0; // type==1:Operand type==0:Operator

    bool sign = 0; // sign==1:negative sign==0:positive

    int tree_height = 0; // tree height for controling parameter to build complete tree

    Node* left = NULL; // left child

    Node* right = NULL; // right child
};

// Operand Categories
inline bool is_strm(std::string op) {
    return (op == "strm1" || op == "strm2" || op == "strm3" || op == "strm4" || op == "-strm1" || op == "-strm2" ||
            op == "-strm3" || op == "-strm4");
}

inline bool is_constant(std::string op) {
    return (op == "c1" || op == "c2" || op == "c3" || op == "c4" || op == "-c1" || op == "-c2" || op == "-c3" ||
            op == "-c4");
}

inline bool is_cell1(std::string op) {
    return (op == "c1" || op == "-c1" || op == "strm1" || op == "-strm1");
}

inline bool is_cell2(std::string op) {
    return (op == "c2" || op == "-c2" || op == "strm2" || op == "-strm2");
}

inline bool is_cell3(std::string op) {
    return (op == "c3" || op == "-c3" || op == "strm3" || op == "-strm3");
}

inline bool is_cell4(std::string op) {
    return (op == "c4" || op == "-c4" || op == "strm4" || op == "-strm4");
}

// Operator Categories
inline bool is_boolean_algebra(std::string op) {
    return (op == "||" || op == "&&" || op == "XOR" || op == "^" || op == "XNOR");
}

inline bool is_comparator(std::string op) {
    return (op == "==" || op == "!=" || op == ">" || op == "<" || op == ">=" || op == "<=");
}

inline bool is_math(std::string op) {
    bool is_mul, is_add;
    is_add = (op == "+" || op == "+  " || op == "+ -" || op == "+- " || op == "+--" || op == "+-" || op == "+ ");
    is_mul = (op == "*" || op == "*  " || op == "* -" || op == "*- " || op == "*--" || op == "*-" || op == "* ");
    return (is_mul || is_add);
}

inline bool is_mux(std::string op) {
    return (op == ":" || op == "?" || op == "[" || op == "]");
}

// build parse tree
inline Node* build_parse_tree(Token& Input_Formula, int target_tree_height) {
    // stack is empty when top==0
    Node* tree_node_stack[16];

    // pointer of stack top
    int top = 0;

    // pointer of Token
    Token* scan;
    scan = Input_Formula.next;

    Node *current, *temp;
    Node *left, *right;

    while (scan != NULL) {
        if (isOperand(scan->data.at(0))) {
            // new tree node && push it into tree stack
            current = new Node;

            // set value
            current->OP = scan->data;
            current->tree_height = 1;

            // operand is negative
            if (scan->data.at(0) == '-') current->sign = 1;

            // type is operand
            current->type = 1;

            // push stack
            top++;
            tree_node_stack[top] = current;
        }

        if (isOperator(scan->data.at(0))) {
            // new root node
            current = new Node;
            current->OP = scan->data;

            // pop two tree node as left and right
            right = tree_node_stack[top];
            top--;
            left = tree_node_stack[top];
            top--;

            // check operand:
            // 1.set elements of cell 1-4 in order
            // 2.put strm as left child, put constant as right child
            if (is_cell1(right->OP) && is_cell2(left->OP)) {
                // swap
                temp = left;
                left = right;
                right = temp;
            } else if (is_cell3(right->OP) && is_cell4(left->OP)) {
                // swap
                temp = left;
                left = right;
                right = temp;
            } else {
                // no action
            }

            if (left->type) {
                if (is_math(current->OP)) {
                    if (left->sign)
                        current->OP += '-';
                    else
                        current->OP += ' ';
                }

                if (is_constant(left->OP)) {
                    // insert a new Node as right mux
                    temp = new Node;
                    temp->OP = "]";
                    temp->tree_height = 2;
                    temp->right = left;
                    left = temp;
                }
            }

            if (right->type) {
                if (is_math(current->OP)) {
                    if (right->sign)
                        current->OP += '-';
                    else
                        current->OP += ' ';
                }

                if (is_strm(right->OP)) {
                    // insert a new Node as left mux
                    temp = new Node;
                    temp->OP = "[";
                    temp->tree_height = 2;
                    temp->left = right;
                    right = temp;
                }
            }

            // check tree height:
            // 1.the left and right tree height should be the same
            // 2.put strm as left child, put constant as right child
            if (left->tree_height > right->tree_height) {
                // if it is constant, insert right, else insert left
                if (right->type && is_constant(right->OP)) {
                    // insert a new Node as right mux, connect right tree
                    temp = new Node;
                    temp->OP = "]";
                    temp->tree_height = right->tree_height + 1;
                    temp->right = right;
                    right = temp;
                } else {
                    // insert a new Node as left mux, connect right tree
                    temp = new Node;
                    temp->OP = "[";
                    temp->tree_height = right->tree_height + 1;
                    temp->left = right;
                    right = temp;
                }
            } else if (left->tree_height < right->tree_height) {
                // if it is constant, insert right, else insert left
                if (left->type && is_constant(left->OP)) {
                    // insert a new Node as right mux, connect left tree
                    temp = new Node;
                    temp->OP = "]";
                    temp->tree_height = left->tree_height + 1;
                    temp->right = left;
                    left = temp;
                } else {
                    // insert a new Node as left mux, connect left tree
                    temp = new Node;
                    temp->OP = "[";
                    temp->tree_height = left->tree_height + 1;
                    temp->left = left;
                    left = temp;
                }
            } else {
                // left->tree_height==right->tree_height
                if (is_cell1(left->OP) && is_cell2(right->OP)) {
                    // left is strm1 while right is c2
                    temp = new Node;
                    temp->OP = "[";
                    temp->tree_height = left->tree_height + 1;
                    temp->left = left;
                    left = temp;

                    temp = new Node;
                    temp->OP = "]";
                    temp->tree_height = right->tree_height + 1;
                    temp->right = right;
                    right = temp;
                } else if (is_cell3(left->OP) && is_cell4(right->OP)) {
                    // left is strm3 while right is c4
                    temp = new Node;
                    temp->OP = "[";
                    temp->tree_height = left->tree_height + 1;
                    temp->left = left;
                    left = temp;

                    temp = new Node;
                    temp->OP = "]";
                    temp->tree_height = right->tree_height + 1;
                    temp->right = right;
                    right = temp;
                } else {
                    // no action
                }
            }

            // connect current with left and right tree
            current->tree_height = left->tree_height + 1;
            current->left = left;
            current->right = right;

            // push stack
            top++;
            tree_node_stack[top] = current;
        }
        scan = scan->next;
    }

    // insert node until tree height==target height
    while (tree_node_stack[top]->tree_height < target_tree_height) {
        // insert a new Node as left mux
        current = new Node;
        current->OP = "[";
        current->tree_height = tree_node_stack[top]->tree_height + 1;
        current->left = tree_node_stack[top];
        tree_node_stack[top] = current;
    }

    return tree_node_stack[top];
}

inline void clear_Tree(Node* tree) {
    if (tree == NULL) {
        return;
    }
    clear_Tree(tree->left);
    clear_Tree(tree->right);
    delete tree;
}

// check the node in parse tree
inline bool check_node(Node* tree, int target_height) {
    bool success = 1;

    if (tree->left == NULL && tree->right == NULL) {
        // check operand
        success &= tree->type;

        // check tree height
        success &= tree->tree_height == target_height;
    } else if (tree->left != NULL && tree->right == NULL) {
        // check operator
        success &= !tree->type;

        // left should be strm
        if (tree->left->type == 1) {
            if (((tree->left->OP == "strm1" || tree->left->OP == "-strm1"))    // for strm1
                || ((tree->left->OP == "strm2" || tree->left->OP == "-strm2")) // for strm2
                || ((tree->left->OP == "strm3" || tree->left->OP == "-strm3")) // for strm3
                || ((tree->left->OP == "strm4" || tree->left->OP == "-strm4")) // for strm4
                // c1, c2, c3, c4 can't be left operand
                )
                success &= 1;
            else
                success &= 0;
        }

        // check operator grammar, in this case, operator should be left mux
        if (tree->OP == "[")
            success &= 1;
        else
            success &= 0;

        // check tree height
        success &= tree->tree_height == target_height;
    } else if (tree->left == NULL && tree->right != NULL) {
        // check operator
        success &= !tree->type;

        // right should be constant
        if (tree->right->type == 1) {
            if (((tree->right->OP == "c1" || tree->right->OP == "-c1"))    // for c1
                || ((tree->right->OP == "c2" || tree->right->OP == "-c2")) // for c2
                || ((tree->right->OP == "c3" || tree->right->OP == "-c3")) // for c3
                || ((tree->right->OP == "c4" || tree->right->OP == "-c4")) // for c4
                // strm1, strm2, strm3, strm4 can't be right operand
                )
                success &= 1;
            else
                success &= 0;
        }

        // check operator grammar, in this case, operator should be right mux
        if (tree->OP == "]")
            success &= 1;
        else
            success &= 0;

        // check tree height
        success &= tree->tree_height == target_height;
    } else if (tree->left != NULL && tree->right != NULL) {
        // check operator
        success &= !tree->type;

        if (tree->left->type == 1 && tree->right->type == 1) {
            // check operand grammar, example: strm1+c4 is forbidden
            if (((tree->left->OP == "strm1" || tree->left->OP == "-strm1") &&
                 (tree->right->OP == "c1" || tree->right->OP == "-c1")) // for strm1
                || ((tree->left->OP == "strm2" || tree->left->OP == "-strm2") &&
                    (tree->right->OP == "c2" || tree->right->OP == "-c2")) // for strm2
                || ((tree->left->OP == "strm3" || tree->left->OP == "-strm3") &&
                    (tree->right->OP == "c3" || tree->right->OP == "-c3")) // for strm3
                || ((tree->left->OP == "strm4" || tree->left->OP == "-strm4") &&
                    (tree->right->OP == "c4" || tree->right->OP == "-c4")) // for strm4
                // c1, c2, c3, c4 can't be left operand
                )
                success &= 1;
            else
                success &= 0;
        } else if (tree->left->type == 0 && tree->right->type == 0) {
            // check operator grammar, example: compare a boolean result with a
            // multiply result is forbidden
            if (is_boolean_algebra(tree->OP)) {
                // if OP is boolean algebra, its left and right must be boolean algebra,
                // comparator or operand
                success &= is_boolean_algebra(tree->left->OP) || is_comparator(tree->left->OP) || tree->left->type;
                success &= is_boolean_algebra(tree->right->OP) || is_comparator(tree->right->OP) || tree->right->type;
            } else if (is_mux(tree->OP)) {
                // when it is '?', the left should be comparator, boolean algebra and
                // mux, while the right should be mux;
                if (tree->OP == "?") {
                    success &= is_comparator(tree->left->OP) || is_boolean_algebra(tree->left->OP);
                    success &= is_mux(tree->right->OP);
                }
            } else {
                // if operator is comparator, math, its left and right must be math
                // compute, mux or operand
                success &= is_math(tree->left->OP) || is_mux(tree->left->OP) || tree->left->type;
                success &= is_math(tree->right->OP) || is_mux(tree->right->OP) || tree->right->type;
            }
        } else
            success &= 0;

        // check tree height
        success &= tree->tree_height == target_height;
    }

    return success;
}

// compare original formula and the transformed formula which stored in parse
// tree
inline bool Parser(Node* tree, int target_height) {
    bool success = 1;

    if (tree != NULL && target_height) {
        // post order visit parse tree
        if (tree->left != NULL) success &= Parser(tree->left, target_height - 1);

        if (tree->right != NULL) success &= Parser(tree->right, target_height - 1);

        success &= check_node(tree, target_height);

#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "Parsing:" << tree->OP << "\n";
#endif
#endif
    }

    return success;
}

//-----------------------------------------------------Hardware OP
// generation--------------------------------------------------------//

// encode operator as an hardware executable code
inline int Operator_Look_Up_Table(std::string Operator) {
    int result;

    // mux
    if (Operator == "?") // 1000
        result = 8;
    if (Operator == ":") // 1010
        result = 10;
    if (Operator == "[") // 1001
        result = 9;
    if (Operator == "]") // 1000
        result = 8;

    // math
    if (Operator == "+") // 0000
        result = 0;
    if (Operator == "+ ") // 0000
        result = 0;
    if (Operator == "+-") // 0001
        result = 1;
    if (Operator == "+  ") // 0000
        result = 0;
    if (Operator == "+ -") // 0001
        result = 1;
    if (Operator == "+- ") // 0010
        result = 2;
    if (Operator == "+--") // 0011
        result = 3;

    if (Operator == "*") // 0100
        result = 4;
    if (Operator == "* ") // 0100
        result = 4;
    if (Operator == "*-") // 0101
        result = 5;
    if (Operator == "*  ") // 0100
        result = 4;
    if (Operator == "* -") // 0101
        result = 5;
    if (Operator == "*- ") // 0101
        result = 5;
    if (Operator == "*--") // 0100
        result = 4;

    // comparator
    if (Operator == ">") // 0000
        result = 0;
    if (Operator == ">=") // 0001
        result = 1;
    if (Operator == "==") // 0010
        result = 2;
    if (Operator == "!=") // 0011
        result = 3;
    if (Operator == "<") // 0100
        result = 4;
    if (Operator == "<=") // 0101
        result = 5;

    // boolean algebra
    if (Operator == "&&") // 1100
        result = 12;
    if (Operator == "||") // 1101
        result = 13;
    if (Operator == "^") // 1110
        result = 14;
    if (Operator == "XOR") // 1110
        result = 14;
    if (Operator == "XNOR") // 1111
        result = 15;

    return result;
}

// generate OP
template <typename Constant_Type1, typename Constant_Type2, typename Constant_Type3, typename Constant_Type4>
bool OP_Generation(
    Node* tree, ap_uint<289>& OP, Constant_Type1 c1, Constant_Type2 c2, Constant_Type3 c3, Constant_Type4 c4) {
    bool success = 1;

    // dynamic ALU cell op1-7
    ap_uint<4> strm_empty, cell_op1, cell_op2, cell_op3, cell_op4, cell_op5, cell_op6, cell_op7;
    ap_uint<1> output_mux;

    // op1-7 in parse tree
    std::string op1, op2, op3, op4, op5, op6, op7, default_op;

    // generate cell1-7 op && strm_empty
    default_op = "+";
    strm_empty = 0;

    if (tree != NULL)
        op7 = tree->OP;
    else
        return !success;

    if (tree->right != NULL) {
        op6 = tree->right->OP;

        if (tree->right->right != NULL) {
            op4 = tree->right->right->OP;
            if (tree->right->right->left == NULL) strm_empty(0, 0) = 1;
        } else {
            op4 = default_op;
            strm_empty(0, 0) = 1;
        }

        if (tree->right->left != NULL) {
            op3 = tree->right->left->OP;
            if (tree->right->left->left == NULL) strm_empty(1, 1) = 1;
        } else {
            op3 = default_op;
            strm_empty(1, 1) = 1;
        }
    } else {
        op6 = default_op;
        op4 = default_op;
        op3 = default_op;
        strm_empty(1, 0) = 3;
    }

    if (tree->left != NULL) {
        op5 = tree->left->OP;

        if (tree->left->right != NULL) {
            op2 = tree->left->right->OP;
            if (tree->left->right->left == NULL) strm_empty(2, 2) = 1;
        } else {
            op2 = default_op;
            strm_empty(2, 2) = 1;
        }

        if (tree->left->left != NULL) {
            op1 = tree->left->left->OP;
            if (tree->left->left->left == NULL) strm_empty(3, 3) = 1;
        } else {
            op1 = default_op;
            strm_empty(3, 3) = 1;
        }
    } else {
        op5 = default_op;
        op2 = default_op;
        op1 = default_op;
        strm_empty(3, 2) = 3;
    }

    cell_op1 = Operator_Look_Up_Table(op1);
    cell_op2 = Operator_Look_Up_Table(op2);
    cell_op3 = Operator_Look_Up_Table(op3);
    cell_op4 = Operator_Look_Up_Table(op4);
    cell_op5 = Operator_Look_Up_Table(op5);
    cell_op6 = Operator_Look_Up_Table(op6);
    cell_op7 = Operator_Look_Up_Table(op7);

    // generate output mux
    if (is_comparator(op7) || is_boolean_algebra(op7))
        output_mux = 1;
    else if (op7 == "?") {
        if ((is_comparator(op3) || is_boolean_algebra(op3)) && (is_comparator(op4) || is_boolean_algebra(op4)))
            output_mux = 1;
        else
            output_mux = 0;
    } else if (op7 == "[") { // choose left
        if (is_comparator(op5) || is_boolean_algebra(op5))
            output_mux = 1;
        else if (op5 == "[") {
            if (is_comparator(op1) || is_boolean_algebra(op1))
                output_mux = 1;
            else
                output_mux = 0;
        } else
            output_mux = 0;
    } else if (op7 == "]" || op7 == ":") // it is impossible in grammar
        return !success;
    else
        output_mux = 0;

    // assign OP
    ap_uint<33> Operator = 0;
    ap_uint<256> Operand = 0;

    Operator(32, 32) = output_mux;
    Operator(31, 28) = strm_empty;
    Operator(27, 24) = cell_op1;
    Operator(23, 20) = cell_op2;
    Operator(19, 16) = cell_op3;
    Operator(15, 12) = cell_op4;
    Operator(11, 8) = cell_op5;
    Operator(7, 4) = cell_op6;
    Operator(3, 0) = cell_op7;

    Operand(255, 192) = c1;
    Operand(191, 128) = c2;
    Operand(127, 64) = c3;
    Operand(63, 0) = c4;

    OP(288, 256) = Operator;
    OP(255, 0) = Operand;

// print for test
#ifndef __SYNTHESIS__
#ifdef DEBUG
    std::cout << "output_mux:" << hex << output_mux << "\n";
    std::cout << "strm_empty:" << hex << strm_empty << "\n";
    std::cout << "op1:" << op1 << "\tcell_op1:" << hex << cell_op1 << "\n";
    std::cout << "op2:" << op2 << "\tcell_op2:" << hex << cell_op2 << "\n";
    std::cout << "op3:" << op3 << "\tcell_op3:" << hex << cell_op3 << "\n";
    std::cout << "op4:" << op4 << "\tcell_op4:" << hex << cell_op4 << "\n";
    std::cout << "op5:" << op5 << "\tcell_op5:" << hex << cell_op5 << "\n";
    std::cout << "op6:" << op6 << "\tcell_op6:" << hex << cell_op6 << "\n";
    std::cout << "op7:" << op7 << "\tcell_op7:" << hex << cell_op7 << "\n";
    std::cout << "c1:" << hex << c1 << "\n";
    std::cout << "c2:" << hex << c2 << "\n";
    std::cout << "c3:" << hex << c3 << "\n";
    std::cout << "c4:" << hex << c4 << "\n";
#endif
#endif

    return success;
}
} // details

/**
 * @brief Generate config bits for dynamic_alu primitive.
 *
 * Notice that caller should free memory allocated in config_bits.
 *
 * @tparam Constant_Type1 type of cosntant1
 * @tparam Constant_Type2 type of cosntant2
 * @tparam Constant_Type3 type of cosntant3
 * @tparam Constant_Type4 type of cosntant4
 *
 * @param expr_str expression std::string.
 * @param input constant1
 * @param input constant2
 * @param input constant3
 * @param input constant4
 * @param generated config bits
 */
template <typename Constant_Type1, typename Constant_Type2, typename Constant_Type3, typename Constant_Type4>
bool dynamicALUOPCompiler(const char expr_str[Formula_Max_Length],
                          Constant_Type1 c1,
                          Constant_Type2 c2,
                          Constant_Type3 c3,
                          Constant_Type4 c4,
                          ap_uint<289>& OP) {
    using namespace xf::database::details;
    bool success = 1;

    Token expr_token;
    Token reverse_polish_token;
    Node* parse_tree;

    // Tokenize
    success &= Pre_Check(expr_str, expr_token);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    if (success) {
        printf("Pre-Check pass!\n");
    } else {
        printf("Generate OP failed! error: pre_check failed\n");
    }
#endif
#endif

    // generate reverse polish notation
    Reverse_Polish(expr_token, reverse_polish_token);

    // generate parse tree
    parse_tree = details::build_parse_tree(reverse_polish_token, Dynamic_ALU_Max_Tree_Height);

    // check grammar by parser
    if (success) success &= details::Parser(parse_tree, Dynamic_ALU_Max_Tree_Height);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    if (success) {
        printf("parser-check pass!\n");
    } else {
        printf("Generate OP failed! error: parser_check failed\n");
    }
#endif
#endif

    // generate hardware op
    if (success)
        success &= OP_Generation<Constant_Type1, Constant_Type2, Constant_Type3, Constant_Type4>(parse_tree, OP, c1, c2,
                                                                                                 c3, c4);

#ifndef __SYNTHESIS__
#ifdef DEBUG
    if (success) {
        printf("OP generated!\n");
    } else {
        printf("Generate OP failed! error: OP not supported\n");
    }
#endif
#endif

    // destroy
    clear_Token(expr_token.next);
    clear_Token(reverse_polish_token.next);
    clear_Tree(parse_tree);

    return success;
}

} // namespace database
} // namespace xf

#endif // XF_DATABASE_DYNAMIC_ALU_HOST_H
