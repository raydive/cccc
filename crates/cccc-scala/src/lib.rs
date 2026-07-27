//! Scala adapter: parses Scala 2 and Scala 3 source with `tree-sitter-scala`
//! and lowers its concrete syntax tree into the language-agnostic
//! [`cccc_core::ir`].
//!
//! Lowering uses an explicit `kind()` dispatch whose default arm recursively
//! visits every named child. Unrecognized Scala syntax is therefore transparent
//! rather than discarded, preserving nested functions and control flow.
//!
//! The adapter maps body-bearing `def`s, lambdas, and partial-function literals
//! to [`Node::Function`], `if` to [`Node::Branch`], `match` and partial-function
//! cases to [`Node::Switch`], loops to [`Node::Loop`], catch handlers to
//! [`Node::Catch`], `&&`/`||` runs to folded [`Node::Logical`] values, and calls
//! to [`Node::Call`]. `try` and `finally` are transparent containers. Abstract
//! `def` declarations are not reported.

use std::path::Path;

use cccc_core::engine;
use cccc_core::ir::{LogicalOp, Node, SwitchCase};
use cccc_core::report::FileReport;
use tree_sitter::Node as TsNode;

/// File extensions analyzed by default.
pub const DEFAULT_EXTS: &[&str] = &["scala"];

/// Parse `source` and produce its scored [`FileReport`].
pub fn analyze_source(path: &Path, source: &str) -> FileReport {
    let (nodes, parse_errors) = to_ir(path, source);
    engine::analyze(&path.display().to_string(), &nodes, parse_errors)
}

/// Parse Scala source and lower it to the shared complexity IR.
pub fn to_ir(_path: &Path, source: &str) -> (Vec<Node>, Vec<String>) {
    let mut parser = tree_sitter::Parser::new();
    if parser
        .set_language(&tree_sitter_scala::LANGUAGE.into())
        .is_err()
    {
        return (Vec::new(), vec!["failed to load Scala grammar".to_string()]);
    }
    let Some(tree) = parser.parse(source, None) else {
        return (Vec::new(), vec!["failed to parse Scala source".to_string()]);
    };

    let mut errors = Vec::new();
    collect_errors(tree.root_node(), source.as_bytes(), &mut errors);

    let mut builder = Builder::new(source.as_bytes());
    builder.visit(tree.root_node());
    (builder.finish(), errors)
}

fn collect_errors(node: TsNode, source: &[u8], out: &mut Vec<String>) {
    if node.is_error() || node.is_missing() {
        let position = node.start_position();
        let mut message = format!(
            "syntax error at line {}, column {}",
            position.row + 1,
            position.column + 1
        );
        if let Some(context) = error_context(node, source) {
            message.push_str(" while parsing ");
            message.push_str(&context);
        }
        if !out.contains(&message) {
            out.push(message);
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_errors(child, source, out);
    }
}

fn error_context(node: TsNode, source: &[u8]) -> Option<String> {
    let mut ancestor = node.parent();
    while let Some(parent) = ancestor {
        let label = match parent.kind() {
            "function_definition" | "function_declaration" => Some("function"),
            "class_definition" => Some("class"),
            "trait_definition" => Some("trait"),
            "object_definition" => Some("object"),
            "enum_definition" => Some("enum"),
            "extension_definition" => Some("extension"),
            "lambda_expression" => Some("lambda"),
            _ => None,
        };
        if let Some(label) = label {
            let name = parent
                .child_by_field_name("name")
                .and_then(|name| name.utf8_text(source).ok())
                .filter(|name| !name.is_empty());
            let subject =
                name.map_or_else(|| label.to_string(), |name| format!("{label} '{name}'"));
            return Some(format!(
                "{subject} starting at line {}",
                parent.start_position().row + 1
            ));
        }
        ancestor = parent.parent();
    }
    None
}

struct Builder<'a> {
    src: &'a [u8],
    stack: Vec<Vec<Node>>,
}

impl<'a> Builder<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            stack: vec![Vec::new()],
        }
    }

    fn finish(mut self) -> Vec<Node> {
        self.stack.pop().expect("module collector")
    }

    fn emit(&mut self, node: Node) {
        self.stack.last_mut().expect("collector").push(node);
    }

    fn collect<F: FnOnce(&mut Self)>(&mut self, walk: F) -> Vec<Node> {
        self.stack.push(Vec::new());
        walk(self);
        self.stack.pop().expect("collector")
    }

    fn text(&self, node: TsNode) -> &str {
        node.utf8_text(self.src).unwrap_or("")
    }

    fn visit_named_children(&mut self, node: TsNode) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if !child.is_extra() {
                self.visit(child);
            }
        }
    }

    fn emit_function_node(&mut self, name: String, kind: &'static str, node: TsNode) {
        let line = node.start_position().row as u32 + 1;
        let body = self.collect(|builder| builder.visit_named_children(node));
        self.emit(Node::Function {
            name,
            kind: kind.to_string(),
            line,
            body,
        });
    }

    fn visit(&mut self, node: TsNode) {
        match node.kind() {
            "function_definition" => {
                let name = node
                    .child_by_field_name("name")
                    .map(|name| self.text(name).to_string())
                    .unwrap_or_else(|| "<function>".to_string());
                let kind = if is_member_definition(node) {
                    "method"
                } else {
                    "function"
                };
                self.emit_function_node(name, kind, node);
            }
            "lambda_expression" => self.emit_function_node("<lambda>".to_string(), "lambda", node),
            "if_expression" => {
                let branch = self.lower_if(node);
                self.emit(branch);
            }
            "while_expression" | "do_while_expression" | "for_expression" => {
                let body = self.collect(|builder| builder.visit_named_children(node));
                self.emit(Node::Loop { body });
            }
            "match_expression" => self.visit_match(node),
            "case_block" | "indented_cases" => self.visit_partial_function(node),
            "try_expression" => self.visit_try(node),
            "infix_expression" => match self.logical_op(node) {
                Some(op) => self.visit_logical(node, op),
                None => self.visit_named_children(node),
            },
            "call_expression" => self.visit_call(node),
            _ => self.visit_named_children(node),
        }
    }

    fn lower_if(&mut self, node: TsNode) -> Node {
        let test = node
            .child_by_field_name("condition")
            .map_or_else(Vec::new, |child| {
                self.collect(|builder| builder.visit(child))
            });
        let then = node
            .child_by_field_name("consequence")
            .map_or_else(Vec::new, |child| {
                self.collect(|builder| builder.visit(child))
            });
        let alternate = node.child_by_field_name("alternative").map(|child| {
            let child = unwrap_single_named_child(child);
            if child.kind() == "if_expression" {
                Box::new(self.lower_if(child))
            } else {
                Box::new(Node::Group(self.collect(|builder| builder.visit(child))))
            }
        });
        Node::Branch {
            test,
            then,
            alternate,
        }
    }

    fn visit_match(&mut self, node: TsNode) {
        if let Some(value) = node.child_by_field_name("value") {
            self.visit(value);
        }

        let cases = node
            .child_by_field_name("body")
            .map_or_else(Vec::new, |body| self.lower_switch_cases(body));
        self.emit(Node::Switch { cases });
    }

    fn visit_partial_function(&mut self, node: TsNode) {
        let line = node.start_position().row as u32 + 1;
        let cases = self.lower_switch_cases(node);
        self.emit(Node::Function {
            name: "<lambda>".to_string(),
            kind: "lambda".to_string(),
            line,
            body: vec![Node::Switch { cases }],
        });
    }

    fn lower_switch_cases(&mut self, node: TsNode) -> Vec<SwitchCase> {
        let mut cases = Vec::new();
        for case in descendant_cases(node) {
            let is_default = case
                .child_by_field_name("pattern")
                .is_some_and(|pattern| pattern.kind() == "wildcard")
                && !named_children(case)
                    .iter()
                    .any(|child| child.kind() == "guard");
            let body = self.collect(|builder| builder.visit_named_children(case));
            cases.push(SwitchCase { is_default, body });
        }
        cases
    }

    fn visit_try(&mut self, node: TsNode) {
        for child in named_children(node) {
            match child.kind() {
                "catch_clause" => self.visit_catch(child),
                _ => self.visit(child),
            }
        }
    }

    fn visit_catch(&mut self, node: TsNode) {
        let cases = descendant_cases(node);
        if cases.is_empty() {
            let body = self.collect(|builder| builder.visit_named_children(node));
            self.emit(Node::Catch { body });
            return;
        }

        for case in cases {
            let body = self.collect(|builder| builder.visit_named_children(case));
            self.emit(Node::Catch { body });
        }
    }

    fn logical_op(&self, node: TsNode) -> Option<LogicalOp> {
        if node.kind() != "infix_expression" {
            return None;
        }
        match node
            .child_by_field_name("operator")
            .map(|operator| self.text(operator))
        {
            Some("&&") => Some(LogicalOp::And),
            Some("||") => Some(LogicalOp::Or),
            _ => None,
        }
    }

    fn visit_logical(&mut self, node: TsNode, op: LogicalOp) {
        let mut operands = Vec::new();
        if let Some(left) = node.child_by_field_name("left") {
            self.collect_logical_side(left, op, &mut operands);
        }
        if let Some(right) = node.child_by_field_name("right") {
            self.collect_logical_side(right, op, &mut operands);
        }
        self.emit(Node::Logical { op, operands });
    }

    fn collect_logical_side(&mut self, side: TsNode, op: LogicalOp, out: &mut Vec<Node>) {
        let side = unwrap_parens(side);
        match self.logical_op(side) {
            Some(side_op) if side_op == op => {
                if let Some(left) = side.child_by_field_name("left") {
                    self.collect_logical_side(left, op, out);
                }
                if let Some(right) = side.child_by_field_name("right") {
                    self.collect_logical_side(right, op, out);
                }
            }
            Some(side_op) => {
                let mut operands = Vec::new();
                if let Some(left) = side.child_by_field_name("left") {
                    self.collect_logical_side(left, side_op, &mut operands);
                }
                if let Some(right) = side.child_by_field_name("right") {
                    self.collect_logical_side(right, side_op, &mut operands);
                }
                out.push(Node::Logical {
                    op: side_op,
                    operands,
                });
            }
            None => out.push(Node::Group(self.collect(|builder| builder.visit(side)))),
        }
    }

    fn visit_call(&mut self, node: TsNode) {
        let callee = node
            .child_by_field_name("function")
            .and_then(|function| self.callee_name(function));
        self.emit(Node::Call { callee });
        self.visit_named_children(node);
    }

    fn callee_name(&self, node: TsNode) -> Option<String> {
        match node.kind() {
            "identifier" | "operator_identifier" => Some(self.text(node).to_string()),
            "field_expression" => node
                .child_by_field_name("field")
                .map(|field| self.text(field).to_string()),
            "generic_function" | "call_expression" => node
                .child_by_field_name("function")
                .and_then(|function| self.callee_name(function)),
            "parenthesized_expression" => named_children(node)
                .into_iter()
                .next()
                .and_then(|inner| self.callee_name(inner)),
            _ => None,
        }
    }
}

fn is_member_definition(node: TsNode) -> bool {
    let mut parent = node.parent();
    while let Some(ancestor) = parent {
        match ancestor.kind() {
            "function_definition" | "lambda_expression" => return false,
            "class_definition"
            | "trait_definition"
            | "object_definition"
            | "enum_definition"
            | "extension_definition" => return true,
            _ => parent = ancestor.parent(),
        }
    }
    false
}

fn unwrap_single_named_child(mut node: TsNode) -> TsNode {
    loop {
        let mut cursor = node.walk();
        let children: Vec<_> = node
            .named_children(&mut cursor)
            .filter(|child| !child.is_extra())
            .collect();
        if children.len() != 1 {
            return node;
        }
        node = children[0];
    }
}

fn unwrap_parens(node: TsNode) -> TsNode {
    if node.kind() == "parenthesized_expression"
        && let [inner] = named_children(node).as_slice()
    {
        return *inner;
    }
    node
}

fn named_children(node: TsNode) -> Vec<TsNode> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter(|child| !child.is_extra())
        .collect()
}

fn descendant_cases(node: TsNode) -> Vec<TsNode> {
    let mut cases = Vec::new();
    for child in named_children(node) {
        if child.kind() == "case_clause" {
            cases.push(child);
        } else if matches!(child.kind(), "case_block" | "indented_cases") {
            cases.extend(
                named_children(child)
                    .into_iter()
                    .filter(|nested| nested.kind() == "case_clause"),
            );
        }
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_source_has_no_functions_or_errors() {
        let report = analyze_source(Path::new("empty.scala"), "");

        assert!(report.functions.is_empty());
        assert!(report.parse_errors.is_empty());
        assert_eq!(report.cognitive, 0);
        assert_eq!(report.cyclomatic, 0);
    }

    #[test]
    fn top_level_def_is_a_function_with_cyclomatic_base() {
        let report = analyze_source(Path::new("answer.scala"), "def answer = 42");

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions.len(), 1);
        let function = &report.functions[0];
        assert_eq!(function.name, "answer");
        assert_eq!(function.kind, "function");
        assert_eq!(function.line, 1);
        assert_eq!(function.cognitive, 0);
        assert_eq!(function.cyclomatic, 1);
    }

    #[test]
    fn definition_in_class_is_a_method() {
        let report = analyze_source(
            Path::new("Service.scala"),
            "class Service:\n  def execute = 42\n",
        );

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions.len(), 1);
        assert_eq!(report.functions[0].name, "execute");
        assert_eq!(report.functions[0].kind, "method");
    }

    #[test]
    fn lambda_is_a_nested_function_unit() {
        let report = analyze_source(
            Path::new("Lambda.scala"),
            "def outer = List(1).map(x => x + 1)\n",
        );

        assert!(report.parse_errors.is_empty());
        let outer = &report.functions[0];
        assert_eq!(outer.children.len(), 1);
        assert_eq!(outer.children[0].name, "<lambda>");
        assert_eq!(outer.children[0].kind, "lambda");
        assert_eq!(outer.children[0].cognitive, 0);
        assert_eq!(outer.children[0].cyclomatic, 1);
        assert_eq!(outer.cognitive, 0);
        assert_eq!(outer.cyclomatic, 1);
    }

    #[test]
    fn partial_function_argument_is_a_nested_lambda_with_a_switch() {
        let source = r#"
def positive(xs: List[Int]) =
  xs.collect {
    case x if x > 0 => x
    case 0 => 0
  }
"#;
        let report = analyze_source(Path::new("Partial.scala"), source);

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        let outer = &report.functions[0];
        assert_eq!(outer.name, "positive");
        assert_eq!(outer.cognitive, 0);
        assert_eq!(outer.cyclomatic, 1);
        assert_eq!(outer.children.len(), 1);
        let partial = &outer.children[0];
        assert_eq!(partial.name, "<lambda>");
        assert_eq!(partial.kind, "lambda");
        assert_eq!(partial.cognitive, 1);
        assert_eq!(partial.cyclomatic, 3);
        assert_eq!(report.cognitive, 1);
        assert_eq!(report.cyclomatic, 4);
    }

    #[test]
    fn partial_function_value_uses_unguarded_wildcard_as_default() {
        let source = r#"
val receive = {
  case Msg(x) => process(x)
  case _ => ignore()
}
"#;
        let report = analyze_source(Path::new("Receive.scala"), source);

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        assert_eq!(report.functions.len(), 1);
        let partial = &report.functions[0];
        assert_eq!(partial.kind, "lambda");
        assert_eq!(partial.cognitive, 1);
        assert_eq!(partial.cyclomatic, 2);
    }

    #[test]
    fn guarded_wildcard_in_partial_function_is_not_default() {
        let source = r#"
val receive = {
  case _ if enabled => process()
}
"#;
        let report = analyze_source(Path::new("GuardedReceive.scala"), source);

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        let partial = &report.functions[0];
        assert_eq!(partial.cognitive, 1);
        assert_eq!(partial.cyclomatic, 2);
    }

    #[test]
    fn control_flow_in_partial_function_case_is_nested_under_switch() {
        let source = r#"
val receive = {
  case Msg(x) =>
    if x > 0 then process(x)
}
"#;
        let report = analyze_source(Path::new("NestedReceive.scala"), source);

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        let partial = &report.functions[0];
        assert_eq!(partial.cognitive, 3);
        assert_eq!(partial.cyclomatic, 3);
    }

    #[test]
    fn indented_partial_function_is_a_nested_lambda() {
        let source = r#"
def positive(xs: List[Int]) =
  xs.collect:
    case x if x > 0 => x
    case _ => 0
"#;
        let report = analyze_source(Path::new("IndentedPartial.scala"), source);

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        let outer = &report.functions[0];
        assert_eq!(outer.children.len(), 1);
        let partial = &outer.children[0];
        assert_eq!(partial.kind, "lambda");
        assert_eq!(partial.cognitive, 1);
        assert_eq!(partial.cyclomatic, 2);
    }

    #[test]
    fn if_expression_adds_one_branch() {
        let report = analyze_source(
            Path::new("Branch.scala"),
            "def choose(flag: Boolean) =\n  if flag then 1\n",
        );

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions[0].cognitive, 1);
        assert_eq!(report.functions[0].cyclomatic, 2);
    }

    #[test]
    fn else_if_chain_is_flat() {
        let report = analyze_source(
            Path::new("ElseIf.scala"),
            "def classify(n: Int) =\n  if n < 0 then -1\n  else if n == 0 then 0\n  else 1\n",
        );

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions[0].cognitive, 3);
        assert_eq!(report.functions[0].cyclomatic, 3);
    }

    #[test]
    fn while_do_while_and_for_are_loops() {
        let report = analyze_source(
            Path::new("Loops.scala"),
            "def loops(xs: List[Int]) = {\n  var n = 0\n  while (n < 1) { n += 1 }\n  do { n -= 1 } while (n > 0)\n  for (x <- xs) { println(x) }\n}\n",
        );

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        assert_eq!(report.functions[0].cognitive, 3);
        assert_eq!(report.functions[0].cyclomatic, 4);
    }

    #[test]
    fn match_counts_non_default_cases() {
        let report = analyze_source(
            Path::new("Match.scala"),
            "def getWords(n: Int) = n match {\n  case 1 => \"one\"\n  case 2 => \"two\"\n  case _ => \"lots\"\n}\n",
        );

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions[0].cognitive, 1);
        assert_eq!(report.functions[0].cyclomatic, 3);
        assert!(report.functions[0].children.is_empty());
    }

    #[test]
    fn try_and_finally_are_transparent_and_each_catch_case_is_a_catch() {
        let source = r#"
def recover() = {
  try { run() }
  catch {
    case _: IOException =>
      if (retryable) retry()
    case _: TimeoutException =>
      timeout()
  }
  finally {
    while (cleaning) { cleanup() }
  }
}
"#;
        let report = analyze_source(Path::new("Recover.scala"), source);

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        assert_eq!(report.functions[0].cognitive, 5);
        assert_eq!(report.functions[0].cyclomatic, 5);
        assert!(report.functions[0].children.is_empty());
    }

    #[test]
    fn logical_runs_fold_and_mixed_operators_nest() {
        let source = r#"
def decide(a: Boolean, b: Boolean, c: Boolean, d: Boolean) = {
  if (a && b && c || d) println("yes")
}
"#;
        let report = analyze_source(Path::new("Logical.scala"), source);

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions[0].cognitive, 3);
        assert_eq!(report.functions[0].cyclomatic, 5);
    }

    #[test]
    fn direct_recursive_call_adds_one_cognitive_point() {
        let source = "def recurse(n: Int): Int = if (n <= 0) 0 else recurse(n - 1)\n";
        let report = analyze_source(Path::new("Recursive.scala"), source);

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions[0].cognitive, 3);
        assert_eq!(report.functions[0].cyclomatic, 2);
    }

    #[test]
    fn abstract_definition_is_not_reported_as_an_executable_function() {
        let source = "trait Service:\n  def execute(input: Int): String\n";
        let report = analyze_source(Path::new("Service.scala"), source);

        assert!(report.parse_errors.is_empty());
        assert!(report.functions.is_empty());
    }

    #[test]
    fn local_definition_is_a_child_function() {
        let source = "def outer = {\n  def inner = 1\n  inner\n}\n";
        let report = analyze_source(Path::new("Local.scala"), source);

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions.len(), 1);
        assert_eq!(report.functions[0].children.len(), 1);
        assert_eq!(report.functions[0].children[0].name, "inner");
        assert_eq!(report.functions[0].children[0].kind, "function");
    }

    #[test]
    fn guarded_wildcard_is_not_a_default_case() {
        let source = "def accept(enabled: Boolean) = 1 match {\n  case _ if enabled => true\n}\n";
        let report = analyze_source(Path::new("Guard.scala"), source);

        assert!(report.parse_errors.is_empty());
        assert_eq!(report.functions[0].cognitive, 1);
        assert_eq!(report.functions[0].cyclomatic, 2);
    }

    #[test]
    fn incomplete_source_reports_parse_errors() {
        let report = analyze_source(Path::new("Broken.scala"), "def broken(");

        assert!(!report.parse_errors.is_empty());
        assert!(report.parse_errors[0].contains("line 1, column"));
    }

    #[test]
    fn parse_error_reports_enclosing_function_context_when_available() {
        let source = "def broken = {\n  if then 1\n}\n";
        let report = analyze_source(Path::new("BrokenContext.scala"), source);

        assert!(!report.parse_errors.is_empty());
        assert!(
            report
                .parse_errors
                .iter()
                .any(|error| error.contains("while parsing function 'broken' starting at line 1"))
        );
    }

    #[test]
    fn for_comprehension_is_one_syntactic_loop_and_guard_is_transparent() {
        let source = "def positives(xs: List[Int]) = for {\n  x <- xs\n  if x > 0\n} yield x\n";
        let report = analyze_source(Path::new("Comprehension.scala"), source);

        assert!(report.parse_errors.is_empty(), "{:?}", report.parse_errors);
        assert_eq!(report.functions[0].cognitive, 1);
        assert_eq!(report.functions[0].cyclomatic, 2);
    }
}
