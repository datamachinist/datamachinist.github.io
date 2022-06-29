---
permalink: /deep_learning/
title: "Deep learning"
excerpt: "Learn about deep neural networks, starting from the basics. "
last_modified_at: 2022-06-28T11:59:26-04:00
toc: true
---

Deep learning is a subfield of machine learning based on the application of deep neural network, i.e. artificial neural networks with a large number of layers. This field has been getting lots of attention lately as more complex problems are being solved with these tools. In this series of posts, we will learn about deep learning and neural networks, starting from the basics.


<!-- Create array of posts with category 'cheat sheet' and sort them alphabetically -->

{% assign sortedPosts = site.categories['deep learning'] | sort: 'title' %}

<!-- Create a list of post using the array defined earlier -->

<ul>
  {% for post in sortedPosts %}
    {% if post.url %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

