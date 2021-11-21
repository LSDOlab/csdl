"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[397],{3905:function(e,t,n){n.d(t,{Zo:function(){return c},kt:function(){return m}});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=a.createContext({}),p=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},c=function(e){var t=p(e.components);return a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},u=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,s=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),u=p(n),m=r,h=u["".concat(s,".").concat(m)]||u[m]||d[m]||i;return n?a.createElement(h,o(o({ref:t},c),{},{components:n})):a.createElement(h,o({ref:t},c))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,o=new Array(i);o[0]=u;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:r,o[1]=l;for(var p=2;p<i;p++)o[p]=n[p];return a.createElement.apply(null,o)}return a.createElement.apply(null,n)}u.displayName="MDXCreateElement"},6061:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return p},toc:function(){return c},default:function(){return u}});var a=n(7462),r=n(3366),i=(n(7294),n(3905)),o=["components"],l={title:"Best Practices"},s=void 0,p={unversionedId:"tutorial/best-practices",id:"tutorial/best-practices",isDocsHomePage:!1,title:"Best Practices",description:"------------------------------------------------------------------------",source:"@site/docs/tutorial/10-best-practices.mdx",sourceDirName:"tutorial",slug:"/tutorial/best-practices",permalink:"/csdl/docs/tutorial/best-practices",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/tutorial/10-best-practices.mdx",tags:[],version:"current",sidebarPosition:10,frontMatter:{title:"Best Practices"},sidebar:"docs",previous:{title:"Advanced",permalink:"/csdl/docs/tutorial/advanced"},next:{title:"Introduction",permalink:"/csdl/docs/examples/intro"}},c=[{value:"Read the error messages",id:"read-the-error-messages",children:[]},{value:"Use the same (compile time) name for the Python Variable object as the (run time) name for the CSDL variable",id:"use-the-same-compile-time-name-for-the-python-variable-object-as-the-run-time-name-for-the-csdl-variable",children:[]},{value:"Prefer Promotion Over Connection",id:"prefer-promotion-over-connection",children:[]},{value:"Don&#39;t redefine variables",id:"dont-redefine-variables",children:[]},{value:"When creating multiple variables in a compile time loop to aggregate later, store them in a list",id:"when-creating-multiple-variables-in-a-compile-time-loop-to-aggregate-later-store-them-in-a-list",children:[]},{value:"Issue connections at the lowest possible level in the model hierarchy",id:"connections",children:[]},{value:"Always assign types to parameters",id:"always-assign-types-to-parameters",children:[]},{value:"Register outputs immediately after they are defined",id:"register-outputs-immediately-after-they-are-defined",children:[]},{value:"Do not append type information to class/object names",id:"do-not-append-type-information-to-classobject-names",children:[]},{value:"Use Python ints, floats, and NumPy arays wherever possible, until you need derivatives",id:"use-python-ints-floats-and-numpy-arays-wherever-possible-until-you-need-derivatives",children:[]}],d={toc:c};function u(e){var t=e.components,n=(0,r.Z)(e,o);return(0,i.kt)("wrapper",(0,a.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("hr",null),(0,i.kt)("p",null,"This page provides some best practices to help you avoid making common\nerrors when using CSDL.\nThis section assumes you have read the preceding sections in this\ntutorial.\nThese best practices are not required, but ",(0,i.kt)("em",{parentName:"p"},"highly recommended"),".\nIf you run into any issues because of something you did wrong, the\n",(0,i.kt)("a",{parentName:"p",href:"../troubleshooting"},"Troubleshooting")," will be more appropriate."),(0,i.kt)("h2",{id:"read-the-error-messages"},"Read the error messages"),(0,i.kt)("p",null,(0,i.kt)("inlineCode",{parentName:"p"},"csdl")," has early error messages meant to prevent users from providing\nill-formed or physically meaningless model specifications.\nThe error messages should be helpful when debugging model\nspecifications.\nIdeally, all errors should occur within ",(0,i.kt)("inlineCode",{parentName:"p"},"Model"),", and if ",(0,i.kt)("inlineCode",{parentName:"p"},"Model")," does\nnot emit an error, then the back end will not emit an error and the\ncompilation will be successful."),(0,i.kt)("h2",{id:"use-the-same-compile-time-name-for-the-python-variable-object-as-the-run-time-name-for-the-csdl-variable"},"Use the same (compile time) name for the Python Variable object as the (run time) name for the CSDL variable"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"lift = self.declare_variable('lift')\n")),(0,i.kt)("p",null,"not"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"l = self.declare_variable('lift')\n")),(0,i.kt)("p",null,"This is not required, but it avoids confusion."),(0,i.kt)("h2",{id:"prefer-promotion-over-connection"},"Prefer Promotion Over Connection"),(0,i.kt)("p",null,"CSDL promotes variable names from child models by default.\nIn some cases, promotions are not allowed, and connections are required.\nPromoting more than one variable from more than one child model is\nallowed if both variables are constructed using\n",(0,i.kt)("inlineCode",{parentName:"p"},"Model.declare_variable"),".\nIf two variables are outputs, either constructed by\n",(0,i.kt)("inlineCode",{parentName:"p"},"Model.create_output"),", or registered using ",(0,i.kt)("inlineCode",{parentName:"p"},"Model.register_output"),", and\nthey have the same name, then they cannot be promoted to the same level\nin the hierarchy."),(0,i.kt)("p",null,"Connections are meant for connecting two variables with different names,\nor at different levels within the model hierarchy."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"self.connect('child_1.a', 'child_2.b')\nself.connect('child_1.c', 'child_2.grandchild.b')\n")),(0,i.kt)("h2",{id:"dont-redefine-variables"},"Don't redefine variables"),(0,i.kt)("p",null,"Python does not enforce variable immutability, so if a variable is redefined (or\nin Python parlance, if a variable name is bound to a new reference),\nthen the object in memory storing the variable data is not in use in\nlater parts of the code."),(0,i.kt)("p",null,"This means that your ",(0,i.kt)("inlineCode",{parentName:"p"},"Model")," will lose track of the original variable.\nEither the original variable will be ignored during dead code removal,\nor the original variable will not be used in other parts of the code\nwhere it should have been, leading to unexpected/incorrect behavior in\nthe final simulation."),(0,i.kt)("h2",{id:"when-creating-multiple-variables-in-a-compile-time-loop-to-aggregate-later-store-them-in-a-list"},"When creating multiple variables in a compile time loop to aggregate later, store them in a list"),(0,i.kt)("p",null,"In Python, you might be used to mutating"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"for i, obj in enumerate(iterable):\n    v[i] += obj\n")),(0,i.kt)("p",null,"use"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"l = []\nfor i, obj in enumerate(iterable):\n    l.append(obj)\ns = csdl.sum(*l)\n")),(0,i.kt)("p",null,"or better yet,"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"x = self.create_ouput('x', shape=(n,))\n# concatenate some variables in x...\ns = csdl.sum(*filter(lambda x: True, [x[i] for i in range(n)]))\n")),(0,i.kt)("h2",{id:"connections"},"Issue connections at the lowest possible level in the model hierarchy"),(0,i.kt)("p",null,"Instead of"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"self.connect('A.B.C.x', 'A.B.y')\n")),(0,i.kt)("p",null,"use"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"self.connect('B.C.x', 'B.y')\n")),(0,i.kt)("p",null,"within the ",(0,i.kt)("inlineCode",{parentName:"p"},"Model")," named ",(0,i.kt)("inlineCode",{parentName:"p"},"'A'"),"."),(0,i.kt)("h2",{id:"always-assign-types-to-parameters"},"Always assign types to parameters"),(0,i.kt)("p",null,"Instead of"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"self.parameters.declare('cubesats')\nself.parameters.declare('groundstations')\nself.parameters.declare('timesteps')\nself.parameters.declare('stepsize')\n")),(0,i.kt)("p",null,"use"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"self.parameters.declare('cubesats', types=dict)\nself.parameters.declare('groundstations', types=dict)\nself.parameters.declare('timesteps', types=int)\nself.parameters.declare('stepsize', types=float)\n")),(0,i.kt)("h2",{id:"register-outputs-immediately-after-they-are-defined"},"Register outputs immediately after they are defined"),(0,i.kt)("p",null,"Prefer"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"a = f(x)\nself.register('a', a)\n")),(0,i.kt)("p",null,"over"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"a = f(x)\n\n# ...\n\nself.register('a', a)\n")),(0,i.kt)("p",null,"This will make your code easier to read (and easier to debug).\nRemember, you can always use a variable after it is registered."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"a = f(x)\nself.register('a', a)\n\n# ...\n\nb = g(a)\n\n# ...\n")),(0,i.kt)("h2",{id:"do-not-append-type-information-to-classobject-names"},"Do not append type information to class/object names"),(0,i.kt)("p",null,"For example,"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"class AirfoilModel(Model):\n    # ...\n")),(0,i.kt)("p",null,"The ",(0,i.kt)("inlineCode",{parentName:"p"},"AirfoilModel")," is a ",(0,i.kt)("inlineCode",{parentName:"p"},"Model")," subclass.\nThis information is already captured in the definition of\n",(0,i.kt)("inlineCode",{parentName:"p"},"AirfoilModel"),".\nInsetead, use ",(0,i.kt)("inlineCode",{parentName:"p"},"Airfoil"),":"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"class Airfoil(Model):\n    # ...\n")),(0,i.kt)("p",null,"There can't be two Python objects named ",(0,i.kt)("inlineCode",{parentName:"p"},"Airfoil")," in the same scope, so\nthere's no risk of confusing what ",(0,i.kt)("inlineCode",{parentName:"p"},"Airfoil")," means in a given context."),(0,i.kt)("h2",{id:"use-python-ints-floats-and-numpy-arays-wherever-possible-until-you-need-derivatives"},"Use Python ints, floats, and NumPy arays wherever possible, until you need derivatives"),(0,i.kt)("p",null,"Python objects used to define a model specification are compile time\nconstants; they are always hardcoded values in the final program (if\nthey are present at all).\nThis provides a performance boost over using CSDL variables because the\nhistory of operations for computing compile time constants (e.g.\n",(0,i.kt)("inlineCode",{parentName:"p"},"2*np.pi"),") is not part of the intermediate representation, so no\nadditional code is generated in the final program."),(0,i.kt)("p",null,"Defining a model specification using entirely Python ints, floats, and\nNumPy arrays however, not only results in a simulation without access to\nderivatives, but this results in no program being generated from CSDL\ncode at all."),(0,i.kt)("p",null,"If you are using CSDL for optimization, or would like to leave that\noption available for code that you initially develop only for running\nanalyses, you will need to use CSDL variables, but wherever you don't\nneed derivatives, compile time constants will give your final code a\nperformance boost."))}u.isMDXComponent=!0}}]);