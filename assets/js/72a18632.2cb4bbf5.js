"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[5780],{3905:function(e,t,n){n.d(t,{Zo:function(){return d},kt:function(){return u}});var a=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var l=a.createContext({}),p=function(e){var t=a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},d=function(e){var t=p(e.components);return a.createElement(l.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},c=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,l=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),c=p(n),u=i,h=c["".concat(l,".").concat(u)]||c[u]||m[u]||o;return n?a.createElement(h,r(r({ref:t},d),{},{components:n})):a.createElement(h,r({ref:t},d))}));function u(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,r=new Array(o);r[0]=c;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:i,r[1]=s;for(var p=2;p<o;p++)r[p]=n[p];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}c.displayName="MDXCreateElement"},5056:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return d},default:function(){return c}});var a=n(7462),i=n(3366),o=(n(7294),n(3905)),r=["components"],s={title:"Variable Types"},l=void 0,p={unversionedId:"tutorial/types",id:"tutorial/types",isDocsHomePage:!1,title:"Variable Types",description:"------------------------------------------------------------------------",source:"@site/docs/tutorial/4-types.mdx",sourceDirName:"tutorial",slug:"/tutorial/types",permalink:"/csdl/docs/tutorial/types",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/tutorial/4-types.mdx",tags:[],version:"current",sidebarPosition:4,frontMatter:{title:"Variable Types"},sidebar:"docs",previous:{title:"Language Concepts",permalink:"/csdl/docs/tutorial/concepts"},next:{title:"Standard Library",permalink:"/csdl/docs/tutorial/std_lib"}},d=[{value:"(Declared) Variable",id:"declared-variable",children:[]},{value:"Input",id:"input",children:[]},{value:"Output",id:"output",children:[]},{value:"Concatenation",id:"concatenation",children:[{value:"For NumPy Users",id:"for-numpy-users",children:[]}]}],m={toc:d};function c(e){var t=e.components,n=(0,i.Z)(e,r);return(0,o.kt)("wrapper",(0,a.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("hr",null),(0,o.kt)("h2",{id:"declared-variable"},"(Declared) Variable"),(0,o.kt)("p",null,"All variables are instances of the ",(0,o.kt)("inlineCode",{parentName:"p"},"Variable")," base class or one of its\nsubclasses.\nIn order to use a variable, it must first be declared using\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.declare_variable"),", which returns a ",(0,o.kt)("inlineCode",{parentName:"p"},"DeclaredVariable")," object."),(0,o.kt)("p",null,"A declared variable represents an input to the model from either a\n",(0,o.kt)("em",{parentName:"p"},"parent model")," or a ",(0,o.kt)("em",{parentName:"p"},"child model"),".\n(Constructing model hierarchies is covered in detail in\nthe section on ",(0,o.kt)("a",{parentName:"p",href:"/docs/tutorial/oo"},"Object Oriented Programming"),".)\nA declared variable is so called because it may or may not be defined\nuntil the model hierarchy is defined."),(0,o.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"Any object of ",(0,o.kt)("inlineCode",{parentName:"p"},"Variable")," class or its subclasses supports establishing a\ndependency on an element or a slice of the array that will be\nconstructed at run time using similar syntax to NumPy."),(0,o.kt)("pre",{parentName:"div"},(0,o.kt)("code",{parentName:"pre",className:"language-py"},"a = b[:9]\n")))),(0,o.kt)("div",{className:"admonition admonition-important alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"important")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"Unlike NumPy, indexing a ",(0,o.kt)("inlineCode",{parentName:"p"},"Variable")," does not automatically remove an axis with\ndimension ",(0,o.kt)("inlineCode",{parentName:"p"},"1"),".\nThis can lead to problems in user code that may be resolved similar to\n",(0,o.kt)("a",{parentName:"p",href:"/docs/troubleshooting#broadcasting"},"how this is resolved"),"."))),(0,o.kt)("p",null,"Each variable in CSDL has a name.\nCSDL variable names are automatically generated unless the user provides\na name by calling ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.declare_variable"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_input"),",\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output"),", or ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.register_output"),"."),(0,o.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"Calls to ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.declare_variable")," need not be at the beginning of the\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.define")," method.\nSince ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.declare_variable")," is used for declaring variables from\nparent or child submodels, it may be used after adding a submodel."))),(0,o.kt)("h2",{id:"input"},"Input"),(0,o.kt)("p",null,"Declared variables provide a way for models in different levels and\nbranches of the model hierarchy to specify data transfers between\ncorresponding parts of a simulation.\nThey do not however, provide a way to enforce that data be transfered\n",(0,o.kt)("em",{parentName:"p"},"into")," the simulation, ",(0,o.kt)("em",{parentName:"p"},"from outside")," the simulation.\nThe ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_input")," method provides a way to signal to CSDL that\ndata ",(0,o.kt)("em",{parentName:"p"},"must")," be provided as an input to the simulation, and not as a data\ntransfer between different parts of a simulation.\nThis is useful for defining optimization problems, where an optimizer\nupdates the ",(0,o.kt)("a",{parentName:"p",href:"/docs/tutorial/optimization"},"design variables"),"."),(0,o.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"An input to the main model in CSDL can be thought of as an entry point\nto the resulting simulation, like an argument to the ",(0,o.kt)("inlineCode",{parentName:"p"},"main")," function in\nC."))),(0,o.kt)("p",null,'CSDL inputs do not have to be created at the top level, or "main" model,\nhowever.\nValues for inputs can be set for a ',(0,o.kt)("inlineCode",{parentName:"p"},"Simulator")," after compile time\nbefore each simulation, regardless of where they are created in the\nhierarchy."),(0,o.kt)("div",{className:"admonition admonition-important alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"important")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"A declared ",(0,o.kt)("inlineCode",{parentName:"p"},"Variable")," is also ",(0,o.kt)("em",{parentName:"p"},"allowed")," to be an input to the\noverall simulation, but an ",(0,o.kt)("inlineCode",{parentName:"p"},"Input")," is ",(0,o.kt)("em",{parentName:"p"},"required"),"  to be an input to the\noverall simulation.\nRemember that an ",(0,o.kt)("inlineCode",{parentName:"p"},"Input")," object represents an input whose source is\n",(0,o.kt)("em",{parentName:"p"},"outside")," the simulation, and is thus an input to the ",(0,o.kt)("em",{parentName:"p"},"entire"),"\nsimulation.\nIn contrast, a declared ",(0,o.kt)("inlineCode",{parentName:"p"},"Variable")," is a data transfer within the\nsimulation at a point corresponding to the ",(0,o.kt)("em",{parentName:"p"},"current")," model, and may be\nan output of a ",(0,o.kt)("em",{parentName:"p"},"parent model"),", ",(0,o.kt)("em",{parentName:"p"},"child model"),", or external code."))),(0,o.kt)("h2",{id:"output"},"Output"),(0,o.kt)("p",null,"Models define variables in terms of other variables.\nThe resulting variable is an output, which is represented by an ",(0,o.kt)("inlineCode",{parentName:"p"},"Output"),"\nobject.\nTo create an output, simply define a new variable using basic\nmathematical expressions or the standard library."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"x = self.declare_variable('x')\ny = csdl.sin(x)\nself.register_output('y', y)\n")),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.register_output")," method provides a way to name an output.\nIn order to ensure that an ",(0,o.kt)("inlineCode",{parentName:"p"},"Output")," generates executable code, it or one\nof its dependencies must be registered as an output."),(0,o.kt)("h2",{id:"concatenation"},"Concatenation"),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output")," returns a ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," for concatenating\nvalues.\nThe ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," class inherits from ",(0,o.kt)("inlineCode",{parentName:"p"},"Output"),"\nAn object of class ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," differs from an ",(0,o.kt)("inlineCode",{parentName:"p"},"Output")," in that an\n",(0,o.kt)("inlineCode",{parentName:"p"},"Output")," is defined in terms of other ",(0,o.kt)("inlineCode",{parentName:"p"},"Variable")," objects, and then\nregistered, but a ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," object is registered first, and then\ndefined in terms of other ",(0,o.kt)("inlineCode",{parentName:"p"},"Variable")," objects.\nThe ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," class also supports multidimensional indexed\nassignment, which behaves the same way that NumPy indexed assignment\ndoes, except that\n",(0,o.kt)("a",{parentName:"p",href:"/docs/troubleshooting#broadcasting"},"broadcasting")," is not supported."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"p = self.create_output('p', shape=(2,))\np[0] = x\np[1] = y\n")),(0,o.kt)("p",null,"In ",(0,o.kt)("a",{parentName:"p",href:"/docs/tutorial/getting-started"},"Getting Started"),", we mentioned that\nCSDL variables are immutable.\nThis is still true of ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," because each index can have at\nmost one assignment.\nThere can be no overlap between indices used in assignment in different\nparts of the code.\nSome indices may be left to their default values."),(0,o.kt)("div",{className:"admonition admonition-important alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"important")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"You don't need to call ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output")," if your ",(0,o.kt)("inlineCode",{parentName:"p"},"Output")," is not a\nconcatenation of variables.\nSee\n",(0,o.kt)("a",{parentName:"p",href:"/docs/troubleshooting#i-get-an-output-not-defined-error-but-i-defined-my-output"},"Troubleshooting"),"."))),(0,o.kt)("div",{className:"admonition admonition-important alert alert--info"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"important")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"When you call ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output"),", you don't need to call\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.register_output")," later.\nSee\n",(0,o.kt)("a",{parentName:"p",href:"/docs/troubleshooting#i-get-an-output-not-defined-error-but-i-defined-my-output"},"Troubleshooting"),"."))),(0,o.kt)("h3",{id:"for-numpy-users"},"For NumPy Users"),(0,o.kt)("p",null,"It's very common to preallocate NumPy variables with ",(0,o.kt)("inlineCode",{parentName:"p"},"np.zeros"),",\n",(0,o.kt)("inlineCode",{parentName:"p"},"np.eye"),", etc. and then assign only some values to a NumPy array.\nThe ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," variable type serves this specific purpose, but it\nhas a slightly different API to maintain immutability."),(0,o.kt)("p",null,"You might be used to defining an array in Python to define a coordinate\nrotation matrix:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"R1 = np.zeros((3, 3))\nR1[0, 0] = 1\nR1[1, 1] = np.cos(theta)\nR1[1, 2] = np.sin(theta)\nR1[2, 1] = -np.sin(theta)\nR1[2, 2] = np.cos(theta)\n")),(0,o.kt)("p",null,"In the above example, ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," is assigned values of ",(0,o.kt)("inlineCode",{parentName:"p"},"0")," at ",(0,o.kt)("em",{parentName:"p"},"run time"),", and\nthen some of those values are ",(0,o.kt)("em",{parentName:"p"},"mutated at run time"),".\nPython allows users to update the values of any of the elments in ",(0,o.kt)("inlineCode",{parentName:"p"},"R1"),"\nfreely.\nCSDL on the other hand, does not.\nIn CSDL, the ",(0,o.kt)("inlineCode",{parentName:"p"},"val")," argument in ",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output")," sets the ",(0,o.kt)("em",{parentName:"p"},"default\nvalue"),", which if not overwritten at compile time, ",(0,o.kt)("em",{parentName:"p"},"remains constant at\nrun time"),".\nIf the value is overwritten at compile time, then the default value ",(0,o.kt)("em",{parentName:"p"},"is\nnever set at run time"),"."),(0,o.kt)("p",null,"Let's take a look at how ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," is defined in CSDL:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"r1_val = np.zeros((3, 3))\nr1_val[0, 0] = 1\nR1 = self.create_output('R1', val=r1_val)\nR1[1, 1] = csdl.cos(theta)\nR1[1, 2] = csdl.sin(theta)\nR1[2, 1] = -csdl.sin(theta)\nR1[2, 2] = csdl.cos(theta)\n")),(0,o.kt)("p",null,"In this example, both ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"theta")," are CSDL variables, which means\nthe history of operations, ",(0,o.kt)("em",{parentName:"p"},"not the run time values")," will be stored in\nthe objects ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," and ",(0,o.kt)("inlineCode",{parentName:"p"},"theta"),".\nThe ",(0,o.kt)("inlineCode",{parentName:"p"},"val")," argument tells CSDL that, unless otherwise specified, the run\ntime value for the array ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," is an array of zeros with shape ",(0,o.kt)("inlineCode",{parentName:"p"},"(3,3)")," (a\n3x3 matrix).\nWe then tell CSDL to store a history of operations for individual\nindices for ",(0,o.kt)("inlineCode",{parentName:"p"},"R1"),"."),(0,o.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"Run time values cannot be assigned directly to an index of a\n",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation"),".\nOnly individual indices of the ",(0,o.kt)("inlineCode",{parentName:"p"},"val")," argument value can be assigned.\nThis is why ",(0,o.kt)("inlineCode",{parentName:"p"},"r_val")," is created and then assigned a constant value, so\nthat ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," will have a value of ",(0,o.kt)("inlineCode",{parentName:"p"},"1")," at the index ",(0,o.kt)("inlineCode",{parentName:"p"},"(0, 0)"),"."))),(0,o.kt)("p",null,"This history of operations define the rotation matrix.\nSince indices are not allowed to overlap, CSDL is guaranteed to assign a\nvalue to each index of ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," exactly ",(0,o.kt)("em",{parentName:"p"},"once")," at run time.\nThe indices ",(0,o.kt)("inlineCode",{parentName:"p"},"(0,1)"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"(0,2)"),", ",(0,o.kt)("inlineCode",{parentName:"p"},"(1,0)"),", and ",(0,o.kt)("inlineCode",{parentName:"p"},"(2,0)")," are left without a\nhistory of operations to define their values at run time, so CSDL will\nassign them whatever values were given in the ",(0,o.kt)("inlineCode",{parentName:"p"},"val")," argument in\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output"),"."),(0,o.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"NumPy array values are still allowed to mutate within\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.define"),", so if you need to construct a constant at compile time,\nyou still have that flexibility because ",(0,o.kt)("em",{parentName:"p"},"NumPy arrays are CSDL compile\ntime constants"),". You can think of compile time constants as hard coded\nvalues in the final program."))),(0,o.kt)("p",null,"Sometimes it's desirable to assign values to a ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," within a\nfunction instead of defining a new ",(0,o.kt)("inlineCode",{parentName:"p"},"Model")," each time.\nA ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," however, cannot be created without a call to the\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output")," method, so a ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," cannot be\ninstantiated within a standalone function."),(0,o.kt)("p",null,"To define a ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," using a function, you will need to pass a\n",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," variable that has no history of operations assigned to\nany of its indices.\nThe example above may be rewritten as follows."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"def rotate_x(R1, theta):\n    R1[1, 1] = csdl.cos(theta)\n    R1[1, 2] = csdl.sin(theta)\n    R1[2, 1] = -csdl.sin(theta)\n    R1[2, 2] = csdl.cos(theta)\n\nr1_val = np.zeros((3, 3))\nr1_val[0, 0] = 1\nR1 = self.create_output('R1', val=r1_val)\nrotate_x(R1, theta)\n")),(0,o.kt)("p",null,"Strictly speaking, ",(0,o.kt)("inlineCode",{parentName:"p"},"rotate_x")," does mutate the ",(0,o.kt)("em",{parentName:"p"},"Python")," object ",(0,o.kt)("inlineCode",{parentName:"p"},"R1"),", but\nconsidering that the definition of a ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," must follow its\nconstruction, and each index of a ",(0,o.kt)("inlineCode",{parentName:"p"},"Concatenation")," may be assigned a\nhistory of operations exactly once, this still follows the policy that\n",(0,o.kt)("em",{parentName:"p"},"CSDL variables")," must be immutable.\nOnce ",(0,o.kt)("inlineCode",{parentName:"p"},"rotate_x")," is called, the definition for ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," is complete, enabling\nmeaningful usage of ",(0,o.kt)("inlineCode",{parentName:"p"},"R1")," in later expressions."),(0,o.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,o.kt)("div",{parentName:"div",className:"admonition-heading"},(0,o.kt)("h5",{parentName:"div"},(0,o.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,o.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,o.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,o.kt)("div",{parentName:"div",className:"admonition-content"},(0,o.kt)("p",{parentName:"div"},"The default value must still be set during the call to\n",(0,o.kt)("inlineCode",{parentName:"p"},"Model.create_output"),".\nIf some elements of a ",(0,o.kt)("inlineCode",{parentName:"p"},"Contatenation")," are not set using CSDL, they will\nretain their default values at run time."))))}c.isMDXComponent=!0}}]);