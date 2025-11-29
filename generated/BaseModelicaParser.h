
// Generated from /home/yacoda/programs/lacemodelica/grammar/BaseModelica.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"


namespace basemodelica {


class  BaseModelicaParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, T__24 = 25, T__25 = 26, 
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, T__30 = 31, T__31 = 32, 
    T__32 = 33, T__33 = 34, T__34 = 35, T__35 = 36, T__36 = 37, T__37 = 38, 
    T__38 = 39, T__39 = 40, T__40 = 41, T__41 = 42, T__42 = 43, T__43 = 44, 
    T__44 = 45, T__45 = 46, T__46 = 47, T__47 = 48, T__48 = 49, T__49 = 50, 
    T__50 = 51, T__51 = 52, T__52 = 53, T__53 = 54, T__54 = 55, T__55 = 56, 
    T__56 = 57, T__57 = 58, T__58 = 59, T__59 = 60, T__60 = 61, T__61 = 62, 
    T__62 = 63, T__63 = 64, T__64 = 65, T__65 = 66, T__66 = 67, T__67 = 68, 
    T__68 = 69, T__69 = 70, T__70 = 71, T__71 = 72, VERSION_HEADER = 73, 
    IDENT = 74, UNSIGNED_NUMBER = 75, UNSIGNED_INTEGER = 76, STRING = 77, 
    WS = 78, LINE_COMMENT = 79, ML_COMMENT = 80
  };

  enum {
    RuleBaseModelica = 0, RuleVersionHeader = 1, RuleClassDefinition = 2, 
    RuleClassPrefixes = 3, RuleClassSpecifier = 4, RuleLongClassSpecifier = 5, 
    RuleShortClassSpecifier = 6, RuleDerClassSpecifier = 7, RuleBasePrefix = 8, 
    RuleEnumList = 9, RuleEnumerationLiteral = 10, RuleComposition = 11, 
    RuleLanguageSpecification = 12, RuleExternalFunctionCall = 13, RuleGenericElement = 14, 
    RuleNormalElement = 15, RuleParameterEquation = 16, RuleGuessValue = 17, 
    RuleBasePartition = 18, RuleSubPartition = 19, RuleClockClause = 20, 
    RuleComponentClause = 21, RuleGlobalConstant = 22, RuleTypePrefix = 23, 
    RuleComponentList = 24, RuleComponentDeclaration = 25, RuleDeclaration = 26, 
    RuleModification = 27, RuleClassModification = 28, RuleArgumentList = 29, 
    RuleArgument = 30, RuleElementModificationOrReplaceable = 31, RuleElementModification = 32, 
    RuleEquation = 33, RuleInitialEquation = 34, RuleStatement = 35, RuleIfEquation = 36, 
    RuleEquationBlock = 37, RuleIfStatement = 38, RuleStatementBlock = 39, 
    RuleForEquation = 40, RuleForStatement = 41, RuleForIndex = 42, RuleWhileStatement = 43, 
    RuleWhenEquation = 44, RuleWhenStatement = 45, RulePrioritizeEquation = 46, 
    RulePrioritizeExpression = 47, RulePriority = 48, RuleDecoration = 49, 
    RuleExpression = 50, RuleExpressionNoDecoration = 51, RuleIfExpression = 52, 
    RuleSimpleExpression = 53, RuleLogicalExpression = 54, RuleLogicalTerm = 55, 
    RuleLogicalFactor = 56, RuleRelation = 57, RuleRelationalOperator = 58, 
    RuleArithmeticExpression = 59, RuleAddOperator = 60, RuleTerm = 61, 
    RuleMulOperator = 62, RuleFactor = 63, RulePrimary = 64, RuleTypeSpecifier = 65, 
    RuleName = 66, RuleComponentReference = 67, RuleFunctionCallArgs = 68, 
    RuleFunctionArguments = 69, RuleFunctionArgumentsNonFirst = 70, RuleArrayArguments = 71, 
    RuleNamedArguments = 72, RuleNamedArgument = 73, RuleFunctionArgument = 74, 
    RuleFunctionPartialApplication = 75, RuleOutputExpressionList = 76, 
    RuleExpressionList = 77, RuleArraySubscripts = 78, RuleSubscript = 79, 
    RuleComment = 80, RuleStringComment = 81, RuleAnnotationComment = 82
  };

  explicit BaseModelicaParser(antlr4::TokenStream *input);

  BaseModelicaParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~BaseModelicaParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class BaseModelicaContext;
  class VersionHeaderContext;
  class ClassDefinitionContext;
  class ClassPrefixesContext;
  class ClassSpecifierContext;
  class LongClassSpecifierContext;
  class ShortClassSpecifierContext;
  class DerClassSpecifierContext;
  class BasePrefixContext;
  class EnumListContext;
  class EnumerationLiteralContext;
  class CompositionContext;
  class LanguageSpecificationContext;
  class ExternalFunctionCallContext;
  class GenericElementContext;
  class NormalElementContext;
  class ParameterEquationContext;
  class GuessValueContext;
  class BasePartitionContext;
  class SubPartitionContext;
  class ClockClauseContext;
  class ComponentClauseContext;
  class GlobalConstantContext;
  class TypePrefixContext;
  class ComponentListContext;
  class ComponentDeclarationContext;
  class DeclarationContext;
  class ModificationContext;
  class ClassModificationContext;
  class ArgumentListContext;
  class ArgumentContext;
  class ElementModificationOrReplaceableContext;
  class ElementModificationContext;
  class EquationContext;
  class InitialEquationContext;
  class StatementContext;
  class IfEquationContext;
  class EquationBlockContext;
  class IfStatementContext;
  class StatementBlockContext;
  class ForEquationContext;
  class ForStatementContext;
  class ForIndexContext;
  class WhileStatementContext;
  class WhenEquationContext;
  class WhenStatementContext;
  class PrioritizeEquationContext;
  class PrioritizeExpressionContext;
  class PriorityContext;
  class DecorationContext;
  class ExpressionContext;
  class ExpressionNoDecorationContext;
  class IfExpressionContext;
  class SimpleExpressionContext;
  class LogicalExpressionContext;
  class LogicalTermContext;
  class LogicalFactorContext;
  class RelationContext;
  class RelationalOperatorContext;
  class ArithmeticExpressionContext;
  class AddOperatorContext;
  class TermContext;
  class MulOperatorContext;
  class FactorContext;
  class PrimaryContext;
  class TypeSpecifierContext;
  class NameContext;
  class ComponentReferenceContext;
  class FunctionCallArgsContext;
  class FunctionArgumentsContext;
  class FunctionArgumentsNonFirstContext;
  class ArrayArgumentsContext;
  class NamedArgumentsContext;
  class NamedArgumentContext;
  class FunctionArgumentContext;
  class FunctionPartialApplicationContext;
  class OutputExpressionListContext;
  class ExpressionListContext;
  class ArraySubscriptsContext;
  class SubscriptContext;
  class CommentContext;
  class StringCommentContext;
  class AnnotationCommentContext; 

  class  BaseModelicaContext : public antlr4::ParserRuleContext {
  public:
    BaseModelicaContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VersionHeaderContext *versionHeader();
    std::vector<antlr4::tree::TerminalNode *> IDENT();
    antlr4::tree::TerminalNode* IDENT(size_t i);
    LongClassSpecifierContext *longClassSpecifier();
    std::vector<ClassDefinitionContext *> classDefinition();
    ClassDefinitionContext* classDefinition(size_t i);
    std::vector<GlobalConstantContext *> globalConstant();
    GlobalConstantContext* globalConstant(size_t i);
    std::vector<DecorationContext *> decoration();
    DecorationContext* decoration(size_t i);
    AnnotationCommentContext *annotationComment();

   
  };

  BaseModelicaContext* baseModelica();

  class  VersionHeaderContext : public antlr4::ParserRuleContext {
  public:
    VersionHeaderContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *VERSION_HEADER();

   
  };

  VersionHeaderContext* versionHeader();

  class  ClassDefinitionContext : public antlr4::ParserRuleContext {
  public:
    ClassDefinitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ClassPrefixesContext *classPrefixes();
    ClassSpecifierContext *classSpecifier();

   
  };

  ClassDefinitionContext* classDefinition();

  class  ClassPrefixesContext : public antlr4::ParserRuleContext {
  public:
    ClassPrefixesContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

   
  };

  ClassPrefixesContext* classPrefixes();

  class  ClassSpecifierContext : public antlr4::ParserRuleContext {
  public:
    ClassSpecifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    LongClassSpecifierContext *longClassSpecifier();
    ShortClassSpecifierContext *shortClassSpecifier();
    DerClassSpecifierContext *derClassSpecifier();

   
  };

  ClassSpecifierContext* classSpecifier();

  class  LongClassSpecifierContext : public antlr4::ParserRuleContext {
  public:
    LongClassSpecifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> IDENT();
    antlr4::tree::TerminalNode* IDENT(size_t i);
    StringCommentContext *stringComment();
    CompositionContext *composition();

   
  };

  LongClassSpecifierContext* longClassSpecifier();

  class  ShortClassSpecifierContext : public antlr4::ParserRuleContext {
  public:
    ShortClassSpecifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENT();
    CommentContext *comment();
    TypeSpecifierContext *typeSpecifier();
    BasePrefixContext *basePrefix();
    ClassModificationContext *classModification();
    EnumListContext *enumList();

   
  };

  ShortClassSpecifierContext* shortClassSpecifier();

  class  DerClassSpecifierContext : public antlr4::ParserRuleContext {
  public:
    DerClassSpecifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> IDENT();
    antlr4::tree::TerminalNode* IDENT(size_t i);
    TypeSpecifierContext *typeSpecifier();
    CommentContext *comment();

   
  };

  DerClassSpecifierContext* derClassSpecifier();

  class  BasePrefixContext : public antlr4::ParserRuleContext {
  public:
    BasePrefixContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

   
  };

  BasePrefixContext* basePrefix();

  class  EnumListContext : public antlr4::ParserRuleContext {
  public:
    EnumListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<EnumerationLiteralContext *> enumerationLiteral();
    EnumerationLiteralContext* enumerationLiteral(size_t i);

   
  };

  EnumListContext* enumList();

  class  EnumerationLiteralContext : public antlr4::ParserRuleContext {
  public:
    EnumerationLiteralContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENT();
    CommentContext *comment();

   
  };

  EnumerationLiteralContext* enumerationLiteral();

  class  CompositionContext : public antlr4::ParserRuleContext {
  public:
    CompositionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<GenericElementContext *> genericElement();
    GenericElementContext* genericElement(size_t i);
    std::vector<BasePartitionContext *> basePartition();
    BasePartitionContext* basePartition(size_t i);
    std::vector<AnnotationCommentContext *> annotationComment();
    AnnotationCommentContext* annotationComment(size_t i);
    std::vector<DecorationContext *> decoration();
    DecorationContext* decoration(size_t i);
    std::vector<EquationContext *> equation();
    EquationContext* equation(size_t i);
    std::vector<InitialEquationContext *> initialEquation();
    InitialEquationContext* initialEquation(size_t i);
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);
    LanguageSpecificationContext *languageSpecification();
    ExternalFunctionCallContext *externalFunctionCall();

   
  };

  CompositionContext* composition();

  class  LanguageSpecificationContext : public antlr4::ParserRuleContext {
  public:
    LanguageSpecificationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STRING();

   
  };

  LanguageSpecificationContext* languageSpecification();

  class  ExternalFunctionCallContext : public antlr4::ParserRuleContext {
  public:
    ExternalFunctionCallContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENT();
    ComponentReferenceContext *componentReference();
    ExpressionListContext *expressionList();

   
  };

  ExternalFunctionCallContext* externalFunctionCall();

  class  GenericElementContext : public antlr4::ParserRuleContext {
  public:
    GenericElementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    NormalElementContext *normalElement();
    ParameterEquationContext *parameterEquation();

   
  };

  GenericElementContext* genericElement();

  class  NormalElementContext : public antlr4::ParserRuleContext {
  public:
    NormalElementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentClauseContext *componentClause();

   
  };

  NormalElementContext* normalElement();

  class  ParameterEquationContext : public antlr4::ParserRuleContext {
  public:
    ParameterEquationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    GuessValueContext *guessValue();
    CommentContext *comment();
    ExpressionContext *expression();
    PrioritizeExpressionContext *prioritizeExpression();

   
  };

  ParameterEquationContext* parameterEquation();

  class  GuessValueContext : public antlr4::ParserRuleContext {
  public:
    GuessValueContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentReferenceContext *componentReference();

   
  };

  GuessValueContext* guessValue();

  class  BasePartitionContext : public antlr4::ParserRuleContext {
  public:
    BasePartitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StringCommentContext *stringComment();
    AnnotationCommentContext *annotationComment();
    std::vector<ClockClauseContext *> clockClause();
    ClockClauseContext* clockClause(size_t i);
    std::vector<SubPartitionContext *> subPartition();
    SubPartitionContext* subPartition(size_t i);

   
  };

  BasePartitionContext* basePartition();

  class  SubPartitionContext : public antlr4::ParserRuleContext {
  public:
    SubPartitionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ArgumentListContext *argumentList();
    StringCommentContext *stringComment();
    AnnotationCommentContext *annotationComment();
    std::vector<EquationContext *> equation();
    EquationContext* equation(size_t i);
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

   
  };

  SubPartitionContext* subPartition();

  class  ClockClauseContext : public antlr4::ParserRuleContext {
  public:
    ClockClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENT();
    ExpressionContext *expression();
    CommentContext *comment();
    DecorationContext *decoration();

   
  };

  ClockClauseContext* clockClause();

  class  ComponentClauseContext : public antlr4::ParserRuleContext {
  public:
    ComponentClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TypePrefixContext *typePrefix();
    TypeSpecifierContext *typeSpecifier();
    ComponentListContext *componentList();

   
  };

  ComponentClauseContext* componentClause();

  class  GlobalConstantContext : public antlr4::ParserRuleContext {
  public:
    GlobalConstantContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TypeSpecifierContext *typeSpecifier();
    DeclarationContext *declaration();
    CommentContext *comment();
    ArraySubscriptsContext *arraySubscripts();

   
  };

  GlobalConstantContext* globalConstant();

  class  TypePrefixContext : public antlr4::ParserRuleContext {
  public:
    TypePrefixContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

   
  };

  TypePrefixContext* typePrefix();

  class  ComponentListContext : public antlr4::ParserRuleContext {
  public:
    ComponentListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ComponentDeclarationContext *> componentDeclaration();
    ComponentDeclarationContext* componentDeclaration(size_t i);

   
  };

  ComponentListContext* componentList();

  class  ComponentDeclarationContext : public antlr4::ParserRuleContext {
  public:
    ComponentDeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DeclarationContext *declaration();
    CommentContext *comment();

   
  };

  ComponentDeclarationContext* componentDeclaration();

  class  DeclarationContext : public antlr4::ParserRuleContext {
  public:
    DeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENT();
    ArraySubscriptsContext *arraySubscripts();
    ModificationContext *modification();

   
  };

  DeclarationContext* declaration();

  class  ModificationContext : public antlr4::ParserRuleContext {
  public:
    ModificationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ClassModificationContext *classModification();
    ExpressionContext *expression();

   
  };

  ModificationContext* modification();

  class  ClassModificationContext : public antlr4::ParserRuleContext {
  public:
    ClassModificationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ArgumentListContext *argumentList();

   
  };

  ClassModificationContext* classModification();

  class  ArgumentListContext : public antlr4::ParserRuleContext {
  public:
    ArgumentListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ArgumentContext *> argument();
    ArgumentContext* argument(size_t i);

   
  };

  ArgumentListContext* argumentList();

  class  ArgumentContext : public antlr4::ParserRuleContext {
  public:
    ArgumentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ElementModificationOrReplaceableContext *elementModificationOrReplaceable();
    DecorationContext *decoration();

   
  };

  ArgumentContext* argument();

  class  ElementModificationOrReplaceableContext : public antlr4::ParserRuleContext {
  public:
    ElementModificationOrReplaceableContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ElementModificationContext *elementModification();

   
  };

  ElementModificationOrReplaceableContext* elementModificationOrReplaceable();

  class  ElementModificationContext : public antlr4::ParserRuleContext {
  public:
    ElementModificationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    NameContext *name();
    StringCommentContext *stringComment();
    ModificationContext *modification();

   
  };

  ElementModificationContext* elementModification();

  class  EquationContext : public antlr4::ParserRuleContext {
  public:
    EquationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    CommentContext *comment();
    SimpleExpressionContext *simpleExpression();
    IfEquationContext *ifEquation();
    ForEquationContext *forEquation();
    WhenEquationContext *whenEquation();
    std::vector<DecorationContext *> decoration();
    DecorationContext* decoration(size_t i);
    ExpressionContext *expression();

   
  };

  EquationContext* equation();

  class  InitialEquationContext : public antlr4::ParserRuleContext {
  public:
    InitialEquationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    EquationContext *equation();
    PrioritizeEquationContext *prioritizeEquation();

   
  };

  InitialEquationContext* initialEquation();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    CommentContext *comment();
    ComponentReferenceContext *componentReference();
    OutputExpressionListContext *outputExpressionList();
    FunctionCallArgsContext *functionCallArgs();
    IfStatementContext *ifStatement();
    ForStatementContext *forStatement();
    WhileStatementContext *whileStatement();
    WhenStatementContext *whenStatement();
    DecorationContext *decoration();
    ExpressionContext *expression();

   
  };

  StatementContext* statement();

  class  IfEquationContext : public antlr4::ParserRuleContext {
  public:
    IfEquationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    std::vector<EquationBlockContext *> equationBlock();
    EquationBlockContext* equationBlock(size_t i);

   
  };

  IfEquationContext* ifEquation();

  class  EquationBlockContext : public antlr4::ParserRuleContext {
  public:
    EquationBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<EquationContext *> equation();
    EquationContext* equation(size_t i);

   
  };

  EquationBlockContext* equationBlock();

  class  IfStatementContext : public antlr4::ParserRuleContext {
  public:
    IfStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    std::vector<StatementBlockContext *> statementBlock();
    StatementBlockContext* statementBlock(size_t i);

   
  };

  IfStatementContext* ifStatement();

  class  StatementBlockContext : public antlr4::ParserRuleContext {
  public:
    StatementBlockContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

   
  };

  StatementBlockContext* statementBlock();

  class  ForEquationContext : public antlr4::ParserRuleContext {
  public:
    ForEquationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ForIndexContext *forIndex();
    std::vector<EquationContext *> equation();
    EquationContext* equation(size_t i);

   
  };

  ForEquationContext* forEquation();

  class  ForStatementContext : public antlr4::ParserRuleContext {
  public:
    ForStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ForIndexContext *forIndex();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

   
  };

  ForStatementContext* forStatement();

  class  ForIndexContext : public antlr4::ParserRuleContext {
  public:
    ForIndexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENT();
    ExpressionContext *expression();

   
  };

  ForIndexContext* forIndex();

  class  WhileStatementContext : public antlr4::ParserRuleContext {
  public:
    WhileStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

   
  };

  WhileStatementContext* whileStatement();

  class  WhenEquationContext : public antlr4::ParserRuleContext {
  public:
    WhenEquationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    std::vector<EquationContext *> equation();
    EquationContext* equation(size_t i);

   
  };

  WhenEquationContext* whenEquation();

  class  WhenStatementContext : public antlr4::ParserRuleContext {
  public:
    WhenStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

   
  };

  WhenStatementContext* whenStatement();

  class  PrioritizeEquationContext : public antlr4::ParserRuleContext {
  public:
    PrioritizeEquationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentReferenceContext *componentReference();
    PriorityContext *priority();

   
  };

  PrioritizeEquationContext* prioritizeEquation();

  class  PrioritizeExpressionContext : public antlr4::ParserRuleContext {
  public:
    PrioritizeExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    PriorityContext *priority();

   
  };

  PrioritizeExpressionContext* prioritizeExpression();

  class  PriorityContext : public antlr4::ParserRuleContext {
  public:
    PriorityContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();

   
  };

  PriorityContext* priority();

  class  DecorationContext : public antlr4::ParserRuleContext {
  public:
    DecorationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *UNSIGNED_INTEGER();

   
  };

  DecorationContext* decoration();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionNoDecorationContext *expressionNoDecoration();
    DecorationContext *decoration();

   
  };

  ExpressionContext* expression();

  class  ExpressionNoDecorationContext : public antlr4::ParserRuleContext {
  public:
    ExpressionNoDecorationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SimpleExpressionContext *simpleExpression();
    IfExpressionContext *ifExpression();

   
  };

  ExpressionNoDecorationContext* expressionNoDecoration();

  class  IfExpressionContext : public antlr4::ParserRuleContext {
  public:
    IfExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionNoDecorationContext *> expressionNoDecoration();
    ExpressionNoDecorationContext* expressionNoDecoration(size_t i);

   
  };

  IfExpressionContext* ifExpression();

  class  SimpleExpressionContext : public antlr4::ParserRuleContext {
  public:
    SimpleExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LogicalExpressionContext *> logicalExpression();
    LogicalExpressionContext* logicalExpression(size_t i);

   
  };

  SimpleExpressionContext* simpleExpression();

  class  LogicalExpressionContext : public antlr4::ParserRuleContext {
  public:
    LogicalExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LogicalTermContext *> logicalTerm();
    LogicalTermContext* logicalTerm(size_t i);

   
  };

  LogicalExpressionContext* logicalExpression();

  class  LogicalTermContext : public antlr4::ParserRuleContext {
  public:
    LogicalTermContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<LogicalFactorContext *> logicalFactor();
    LogicalFactorContext* logicalFactor(size_t i);

   
  };

  LogicalTermContext* logicalTerm();

  class  LogicalFactorContext : public antlr4::ParserRuleContext {
  public:
    LogicalFactorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    RelationContext *relation();

   
  };

  LogicalFactorContext* logicalFactor();

  class  RelationContext : public antlr4::ParserRuleContext {
  public:
    RelationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ArithmeticExpressionContext *> arithmeticExpression();
    ArithmeticExpressionContext* arithmeticExpression(size_t i);
    RelationalOperatorContext *relationalOperator();

   
  };

  RelationContext* relation();

  class  RelationalOperatorContext : public antlr4::ParserRuleContext {
  public:
    RelationalOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

   
  };

  RelationalOperatorContext* relationalOperator();

  class  ArithmeticExpressionContext : public antlr4::ParserRuleContext {
  public:
    ArithmeticExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<TermContext *> term();
    TermContext* term(size_t i);
    std::vector<AddOperatorContext *> addOperator();
    AddOperatorContext* addOperator(size_t i);

   
  };

  ArithmeticExpressionContext* arithmeticExpression();

  class  AddOperatorContext : public antlr4::ParserRuleContext {
  public:
    AddOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

   
  };

  AddOperatorContext* addOperator();

  class  TermContext : public antlr4::ParserRuleContext {
  public:
    TermContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<FactorContext *> factor();
    FactorContext* factor(size_t i);
    std::vector<MulOperatorContext *> mulOperator();
    MulOperatorContext* mulOperator(size_t i);

   
  };

  TermContext* term();

  class  MulOperatorContext : public antlr4::ParserRuleContext {
  public:
    MulOperatorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

   
  };

  MulOperatorContext* mulOperator();

  class  FactorContext : public antlr4::ParserRuleContext {
  public:
    FactorContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<PrimaryContext *> primary();
    PrimaryContext* primary(size_t i);

   
  };

  FactorContext* factor();

  class  PrimaryContext : public antlr4::ParserRuleContext {
  public:
    PrimaryContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *UNSIGNED_NUMBER();
    antlr4::tree::TerminalNode *STRING();
    FunctionCallArgsContext *functionCallArgs();
    ComponentReferenceContext *componentReference();
    OutputExpressionListContext *outputExpressionList();
    ArraySubscriptsContext *arraySubscripts();
    std::vector<ExpressionListContext *> expressionList();
    ExpressionListContext* expressionList(size_t i);
    ArrayArgumentsContext *arrayArguments();

   
  };

  PrimaryContext* primary();

  class  TypeSpecifierContext : public antlr4::ParserRuleContext {
  public:
    TypeSpecifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    NameContext *name();

   
  };

  TypeSpecifierContext* typeSpecifier();

  class  NameContext : public antlr4::ParserRuleContext {
  public:
    NameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> IDENT();
    antlr4::tree::TerminalNode* IDENT(size_t i);

   
  };

  NameContext* name();

  class  ComponentReferenceContext : public antlr4::ParserRuleContext {
  public:
    ComponentReferenceContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> IDENT();
    antlr4::tree::TerminalNode* IDENT(size_t i);
    std::vector<ArraySubscriptsContext *> arraySubscripts();
    ArraySubscriptsContext* arraySubscripts(size_t i);

   
  };

  ComponentReferenceContext* componentReference();

  class  FunctionCallArgsContext : public antlr4::ParserRuleContext {
  public:
    FunctionCallArgsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FunctionArgumentsContext *functionArguments();

   
  };

  FunctionCallArgsContext* functionCallArgs();

  class  FunctionArgumentsContext : public antlr4::ParserRuleContext {
  public:
    FunctionArgumentsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    FunctionArgumentsNonFirstContext *functionArgumentsNonFirst();
    ForIndexContext *forIndex();
    FunctionPartialApplicationContext *functionPartialApplication();
    NamedArgumentsContext *namedArguments();

   
  };

  FunctionArgumentsContext* functionArguments();

  class  FunctionArgumentsNonFirstContext : public antlr4::ParserRuleContext {
  public:
    FunctionArgumentsNonFirstContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FunctionArgumentContext *functionArgument();
    FunctionArgumentsNonFirstContext *functionArgumentsNonFirst();
    NamedArgumentsContext *namedArguments();

   
  };

  FunctionArgumentsNonFirstContext* functionArgumentsNonFirst();

  class  ArrayArgumentsContext : public antlr4::ParserRuleContext {
  public:
    ArrayArgumentsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    ForIndexContext *forIndex();

   
  };

  ArrayArgumentsContext* arrayArguments();

  class  NamedArgumentsContext : public antlr4::ParserRuleContext {
  public:
    NamedArgumentsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<NamedArgumentContext *> namedArgument();
    NamedArgumentContext* namedArgument(size_t i);

   
  };

  NamedArgumentsContext* namedArguments();

  class  NamedArgumentContext : public antlr4::ParserRuleContext {
  public:
    NamedArgumentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENT();
    FunctionArgumentContext *functionArgument();

   
  };

  NamedArgumentContext* namedArgument();

  class  FunctionArgumentContext : public antlr4::ParserRuleContext {
  public:
    FunctionArgumentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FunctionPartialApplicationContext *functionPartialApplication();
    ExpressionContext *expression();

   
  };

  FunctionArgumentContext* functionArgument();

  class  FunctionPartialApplicationContext : public antlr4::ParserRuleContext {
  public:
    FunctionPartialApplicationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    TypeSpecifierContext *typeSpecifier();
    NamedArgumentsContext *namedArguments();

   
  };

  FunctionPartialApplicationContext* functionPartialApplication();

  class  OutputExpressionListContext : public antlr4::ParserRuleContext {
  public:
    OutputExpressionListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);

   
  };

  OutputExpressionListContext* outputExpressionList();

  class  ExpressionListContext : public antlr4::ParserRuleContext {
  public:
    ExpressionListContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);

   
  };

  ExpressionListContext* expressionList();

  class  ArraySubscriptsContext : public antlr4::ParserRuleContext {
  public:
    ArraySubscriptsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<SubscriptContext *> subscript();
    SubscriptContext* subscript(size_t i);

   
  };

  ArraySubscriptsContext* arraySubscripts();

  class  SubscriptContext : public antlr4::ParserRuleContext {
  public:
    SubscriptContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();

   
  };

  SubscriptContext* subscript();

  class  CommentContext : public antlr4::ParserRuleContext {
  public:
    CommentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StringCommentContext *stringComment();
    AnnotationCommentContext *annotationComment();

   
  };

  CommentContext* comment();

  class  StringCommentContext : public antlr4::ParserRuleContext {
  public:
    StringCommentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> STRING();
    antlr4::tree::TerminalNode* STRING(size_t i);

   
  };

  StringCommentContext* stringComment();

  class  AnnotationCommentContext : public antlr4::ParserRuleContext {
  public:
    AnnotationCommentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ClassModificationContext *classModification();

   
  };

  AnnotationCommentContext* annotationComment();


  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

}  // namespace basemodelica
