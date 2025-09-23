## agently使用示例
from agently import Agently

agent = Agently.create_agent()

(
    agent
        .input("What time is it now?", always=True)
        .info({
            "default_timezone": "",
            "tool_list": [{
                "name": "get_current_time",
                "desc": "Get current time by time zone provided",
                "kwargs": {
                    "timezone_str": (str, "time zone string in ZoneInfo()"),
                },
            }]
        })
        .output({
            "first_time_response": (str, ),
            "tool_using_judgement": (bool, ),
            "tool_using_command": (
                {
                    "name": (str, "Decide which tool to use by tool name:{tool_list.[].name}"),
                    "kwargs": (dict, "According {tool_list.[].args} to output kwargs dictionary"),
                },
                "If {tool_using_judgement}==False, just output {}",
            ),
        })
)
使用Output输出结构表达语法格式
在代码开发过程中，我们往往需要使用结构化的数据而非自然松散的字符串文本来进行信息存储和传递。这类结构化的数据往往还不是只有一个层级的简单字典(dict)或是只存放一些字符串的列表(list)，而是一种更加复杂综合的结构。

例如，在上面的例子中，我们获得的结果是这样的：


{ 'reply': '在Python中...',
  'next_questions': [
      '我能举一个使用协程的简单例子吗？',
      'Python中的async和await关键字是如何工作的？',
      '如果我想深入了解线程，有哪些好的学习资源或实践项目推荐？'
  ],
}
这是一个复杂的字典结构，通过Output输出结构表达语法，我们可以直接从请求中使用result = agent....start()变量赋值的方式回收这个字典结果，并且用result["reply"]等方式直接使用字典中的字段值。
要实现这样的输出效果，我们需要让模型理解，在生成结果的时候，我们需要得到一个具有两个字段的字典结果。其中在reply字段中，我们需要得到一个长文本字符串，用于存放对用户问题的回复答案。同时，在next_questions字段中，我们需要得到一个列表，用于存放针对本次问答可以进一步提出的问题清单，且我们希望这个问题清单里的问题不要少于3个。

那么我们在使用Agently框架的Output输出结构表达语法时，应该如何思考呢？

首先，确定期望输出的数据结构：

按照上面的描述，我们期望得到的数据结构如下：


{
    "reply": str,
    "next_questions": [str]
}
如果我们将str、int、bool、float等类型的数值看作输出中的带有具体内容的值节点，那么上面这个结构则表达了我们希望输出的结构特征。

接下来，我们使用输出结构表达语法中元组(tuple)的特殊用法来表达对值节点的输出期望：

因为在输出的数据中，我们几乎不会用到元组(tuple)这种数据结构，因此Agently框架赋予了元组新的含义：通过("<类型描述>", "<输出内容期望>")的格式，描述对具体内容的值节点的输出期望。

例如，我们希望在reply节点中获得对本次提问的直接回复，我们就可以做如下表达：


# ("<类型描述>", "<输出内容期望>")
("str", "对本次提问的直接回复")
如果我们希望做更明确的信息指向，比如希望对“本次提问”到底指的是哪部分信息做出明确指向，我们可以使用{}进行指向标注，如果我们希望明确表达这个具体的信息是从哪部分信息块中来的，可以使用{信息块.具体字段名}的方式进行表达：


("str", "对{input.question}的直接回复")
可能也有人注意到，有时候我们需要对list结构做一些额外的说明，比如约定list结构中输出的结果的数量，我们也可以将结构嵌入元组表达中，例如：


([], "最少输出3个结果")
而在元组中的结构，还可以继续嵌入元组表达：


([("str", "根据{reply}可以进一步提出的问题")], "最少输出3个结果")
最后，整合上面两步，形成完整的输出结构表达，并放入.output()请求中：


.output({
    "reply": ("str", "对{input.question}的直接回复"),
    "next_questions": (
        [ ("str", "根据{reply}可以进一步提出的问题") ],
        "最少输出3个结果"
    ),
})

——————————————————流式结构化输出方法
import datetime
import Agently
agent = (
    Agently.create_agent()
        #.set_setting(...)
)

# 使用监听器监听新引入的instant事件
@agent.on_event("instant")
def instant_handler(data):
    # 返回的事件数据结构：
    # `key`: <str> 当前正在输出的键（采用Agently Instant表达方法）
    # `indexes`: <list> 如果当前正在输出的键路径中存在数组，`indexes`里会提供当前输出
    #                   是路径中数组的第几项
    # `delta`: <any> 当前正在输出的键值，如果键值类型是str，此字段更新每次添加的新内容
    #                否则只在键值完全生成完毕后抛出事件，此时字段值和`value`字段值一致
    # `value`: <any> 当前正在输出的键值，如果键值类型是str，此字段更新当前已生成的全量值
    #                否则只在键值完全生成完毕后抛出事件，此时字段值和`delta`字段值一致
    # `complete_value`: <any> 在当前事件抛出时，已经输出的结构化数据的全量内容

    # 输出Instant模式过程结果和输出时间
    print(datetime.now(), data["key"], data["indexes"], data["delta"])

result = (
    agent
        # 使用.use_instant()开启instant模式
        # 3.4.0.3版本之后可以省去此步
        .use_instant()
        .input("Generate 3 other words, then use those 3 words to make a sentence, then generate 4 numbers.")
        # 使用Agently Output语法定义一个复杂结构数据
        .output({
            "words": [("str", )],
            "sentence": ("str", ),
            "numbers": [{ "value": ("int", ) }]
        })
        .start()
)
# 输出最终结果和完成时间
print(datetime.now(), result)

Instant模式输出：
2024-11-03 02:20:01.650752 words.[].$delta [0] cat
2024-11-03 02:20:01.831325 words.[].$delta [1] mouse
2024-11-03 02:20:01.835427 words.[] [0] cat
2024-11-03 02:20:01.849140 words.[].$delta [2] run
2024-11-03 02:20:01.850624 words.[] [1] mouse
2024-11-03 02:20:01.912867 words [] ['cat', 'mouse', 'run']
2024-11-03 02:20:01.913157 words.[] [2] run
2024-11-03 02:20:01.962901 sentence.$delta [] The
2024-11-03 02:20:01.980559 sentence.$delta []  cat
2024-11-03 02:20:01.998184 sentence.$delta []  chased
2024-11-03 02:20:02.015376 sentence.$delta []  the
2024-11-03 02:20:02.032466 sentence.$delta []  mouse
2024-11-03 02:20:02.050336 sentence.$delta []  as
2024-11-03 02:20:02.088583 sentence.$delta []  it
2024-11-03 02:20:02.091482 sentence.$delta []  ran
2024-11-03 02:20:02.102013 sentence.$delta []  for
2024-11-03 02:20:02.118886 sentence.$delta []  its
2024-11-03 02:20:02.136612 sentence.$delta []  life
2024-11-03 02:20:02.154099 sentence.$delta [] .
2024-11-03 02:20:02.258635 sentence [] The cat chased the mouse as it ran for its life.
2024-11-03 02:20:02.556008 numbers.[] [0] {'value': 123}
2024-11-03 02:20:02.556662 numbers.[].value [0] 123
2024-11-03 02:20:02.747380 numbers.[] [1] {'value': 456}
2024-11-03 02:20:02.748144 numbers.[].value [1] 456
2024-11-03 02:20:02.938182 numbers.[] [2] {'value': 789}
2024-11-03 02:20:02.938688 numbers.[].value [2] 789
2024-11-03 02:20:03.483925  [] {'words': ['cat', 'mouse', 'run'], 'sentence': 'The cat chased the mouse as it ran for its life.', 'numbers': [{'value': 123}, {'value': 456}, {'value': 789}, {'value': 101112}]}
2024-11-03 02:20:03.484688 numbers [] [{'value': 123}, {'value': 456}, {'value': 789}, {'value': 101112}]
2024-11-03 02:20:03.485579 numbers.[] [3] {'value': 101112}
2024-11-03 02:20:03.486465 numbers.[].value [3] 101112

最终Result：
2024-11-03 02:20:03.490869 {'words': ['cat', 'mouse', 'run'], 'sentence': 'The cat chased the mouse as it ran for its life.', 'numbers': [{'value': 123}, {'value': 456}, {'value': 789}, {'value': 101112}]}

# 下面复杂数据结构中，键值字符串即为`key`和`indexes`的值，用|进行分割
{
    "value_a": "value_a | []",
    "dict_a": {
        "key_1": "dict_a.key_1 | []",
        "list_in_dict_a": [
            "dict_a.list_in_dict_a.[] | [0]",
            "dict_a.list_in_dict_a.[] | [1]",
            ...
        ],
        "list_with_dict_in_dict_a": [
            {
                "key_2": "dict_a.list_with_dict_in_dict_a.[].key_2 | [0]"
            },
            ...
        ]
    },
    "list_a": [
        "list_a.[] | [0]",
        "list_a.[] | [1]",
        ...
    ],
    "list_b": [
        {
            "list_with_dict_in_list_b": [
                {
                    "key_3": "list_b.[].list_with_dict_in_list_b.[].key_3 | [0, 0]"
                },
                {
                    "key_3": "list_b.[].list_with_dict_in_list_b.[].key_3 | [0, 1]"
                },
                ...
            ]
        },
        {
            "list_with_dict_in_list_b": [
                {
                    "key_3": "list_b.[].list_with_dict_in_list_b.[].key_3 | [1, 0]"
                },
                {
                    "key_3": "list_b.[].list_with_dict_in_list_b.[].key_3 | [1, 1]"
                },
                ...
            ]
        },
    ]
}

通过理解上面的键表达语法，您就可以使用 Agently Instant 方案，通过在instant事件监听器中加入条件过滤的方式，更加实时地获取和处理目标键内容。

例如如果想要从上面的案例中获取key_3（假设它是个字符串类型的键）的实时更新内容，您只需要在监听器中这样写：
@agent.on_event("instant")
def instant_handler(data):
    if data["key"] == "list_b.[].list_with_dict_in_list_b.[].key_3":
        print(data["delta"])
    elif ...:
        ...
    else ...:
        ...

更进一步，如果仅仅想要list_b全部元素中，list_with_dict_in_list_b的第一个元素里的key_3的值，您只需要在监听器中这样写：
@agent.on_event("instant")
def instant_handler(data):
    if (
        data["key"] == "list_b.[].list_with_dict_in_list_b.[].key_3"
        and data["indexes"][1] == 0
    ):
        print(data["delta"])
    elif ...:
        ...
    else ...:
        ...

再简单点，直接使用事件监听器监听特定键
v3.4.0.3更新：看完上面的开发方法之后，可能有的开发者会提出这样的问题：如果我只关心特定的少数键值的监听，为什么还要处理instant事件抛出的所有数据？有没有更简单的定点监听表达方式？

当然有， Agently Instant 方案也为开发者提供了instant:<key_expression>的监听表达方式。

同样用上面的案例，获取key_3的实时更新内容，您还可以这样写：


@agent.on_event("instant:list_b.[].list_with_dict_in_list_b.[].key_3")
def instant_handler(data):
    print(data["delta"])
更进一步，如果仅仅想要list_b全部元素中，list_with_dict_in_list_b的第一个元素里的key_3的值，您还可以这样写：


@agent.on_event("instant:list_b.[].list_with_dict_in_list_b.[].key_3?_,0")
其中通过?分割key和indexes内容，在indexes内容中，如果需要输入多个元素定位要求，可以通过,进行分割，其中_（或者*）表示接受该位置的所有元素，您也可以通过(0|2|4)的方式表达接受该位置的多个元素。

v3.4.0.4更新：再进一步，如果您希望监听器同时处理多个条件，可以使用&对多个条件进行组合：


@agent.on_event("instant:value_a&dict_a.key_1&list_b.[].list_with_dict_in_list_b.[].key_3?_,0")
跟上行业开发习惯，用Generator也可以输出流式事件
当然了，在其他行业工具中，流式输出往往会结合Generator一起使用，比如Gradio就是一个典型例子，如果要使用流式更新，就需要向它传递可以被for循环进行逐项轮询的Generator实例。在v3.4.0.3版本的更新中，Agently Instant 也带来了适配Generator的输出方案，您只需要将.start()换成.get_instant_generator()即可获取到包含所有instant事件的Generator输出实例了，示例代码如下：


generator = (
    agent
        .input("Generator 10 sentences")
        .output({
            "sentences": ([("str", )]),
        })
        .get_instant_generator()
)

for item in generator:
    print(item["key"], item["delta"])
Generator也可以指定监听的key和indexes
v3.4.0.4更新：在监听器中只针对特定事件的监听，在Generator中也可以做到：


# 监听器表达
@agent.on_event("instant:value_a&dict_a.key_1&list_b.[].list_with_dict_in_list_b.[].key_3?_,0")
def handler(data):
    pass

# Generator表达
generator = agent.get_instant_keys_generator("instant:value_a&dict_a.key_1&list_b.[].list_with_dict_in_list_b.[].key_3?_,0")

for item in generator:
    print(item["key"], item["delta"])