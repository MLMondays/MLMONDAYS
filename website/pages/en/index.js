/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="projectLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = props => (
      <h2 className="projectTitle">
        {props.title}
        <small>{props.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={`${baseUrl}img/android-chrome-192x192.png`} />
        <div className="inner">
          <ProjectTitle tagline={siteConfig.tagline} title={siteConfig.title} />
          <PromoSection>
            <Button href={docUrl('doc1.html')}>Docs</Button>
            <Button href={docUrl('doc2.html')}>Data</Button>
            <Button href={docUrl('doc3.html')}>Help</Button>
            <Button href={"https://github.com/dbuscombe-usgs/DL-CDI2020"}>Github Repository</Button>
            <Button href={"https://forms.office.com/Pages/ResponsePage.aspx?id=urWTBhhLe02TQfMvQApUlAxdiRifVmlAg0g-PN54QUVUQVJKUjRDM0pNWk5UUVBaOFdQUE9IRUVISiQlQCN0PWcu"}>Register your interest</Button>

          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}


class HomeSplash2 extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <div className="inner">
          <PromoSection>
            <Button href={"https://forms.office.com/Pages/ResponsePage.aspx?id=urWTBhhLe02TQfMvQApUlAxdiRifVmlAg0g-PN54QUVUQVJKUjRDM0pNWk5UUVBaOFdQUE9IRUVISiQlQCN0PWcu"}>Register your interest</Button>

          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}


class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const FeatureCallout = () => (
      <div
        className="productShowcaseSection paddingBottom"
        style={{textAlign: 'center'}}>
        <h2>Course Overview</h2>
        <MarkdownBlock></MarkdownBlock>
      </div>
    );

    const ImClass = () => (
      <Block background="dark">
        {[
          {
            content:
              'In the last week we will cover some more advanced and emerging techniques for image classification. '+
              'Many approaches in machine learning require a measure of distance between data points (Euclidean, City-Block, Cosine, etc.).'+
              'Here we use a "weakly supervised" deep learning framework based on distance metrics, '+
              'using the concept of maximizing the distance between classes in embedding space.'+
              '<br/><br/> The image to the right shows the position of sample images (black dots) within an embedding space '+
              'from a deep learning model training to identify landuse/landcover (LULC). This approach is designed to improve upon '+
              'supervised image recognition (such as in week 1) in two ways:'+
              '<br/><br/> 1. they potentially require less data to train, and'+
              '<br/><br/> 2. they provide a sample metric that can be used a goodness-of-fit measure',
            image: `${baseUrl}img/metriclearning.png`,
            imageAlign: 'right',
            title: 'Week 4: Semi-supervised image classification',
          },
        ]}
      </Block>
    );


    const ImSeg = () => (
      <Block background="light">
        {[
          {
            content:
              'The images on the left show some examples of image segmentation. '+
              'The vegetation in the images in the far left column are segmented to form a binary '+
              'mask, where white is vegetation and black is everything else (center column). '+
              'The segmented images (the right column) shows the original image segmented with the mask. '+
              '<br/><br/>We can also estimate multiple classes at once. The models we use to do this '+
              'need to be trained using lots of examples of images and their associated label images. '+
              'To deal with the large intra-class variability, which is often implied in natural landcovers/uses, '+
              'we require a powerful machine learning model to '+
              'carry out the segmentation. We will use another type of deep convolutional neural network, '+
              'this time configured to be an autoencoder-decoder network based on the U-Net.'+
              '<br/><br/> This is useful for things like:'+
              '<br/><br/> * quantifying the spatial extent of objects and features of interest'+
              '<br/><br/> * quantifying everything in the scene as a unique class, with no features or objects unlabelled',
            image: `${baseUrl}img/seg_ex.png`,
            imageAlign: 'left',
            title: 'Week 3: Image segmentation',
          },
        ]}
      </Block>
    );

    const ObjRecog = () => (
      <Block background="dark">
        {[
          {
            content:
              'The image on the right shows an example of object recognition, which is '+
              'the detection and localization of objects (in this case, people on a beach). Localization means the model can draw '+
              'a rectange or "bounding box" around each object of each class in each image. '+
              'It answers the question, "where is this thing in this image?"'+
              '<br/><br/> This is useful for things like:'+
              '<br/><br/> * counting people on beaches, counting animals and birds in static cameras, etc, etc'+
              '<br/><br/> * quantifying the proximity of detected objects to other important objects in the same scene ',
            image: `${baseUrl}img/objrecog.png`,
            imageAlign: 'right',
            title: 'Week 2: Object recognition',
          },
        ]}
      </Block>
    );

    const ImRecog = () => (
      <Block background="light">
        {[
          {
            content:
            'The image on the left shows an example of image "recognition", which is classification '+
            'of the entire image, rather than individual pixels. It answers the question, "is this thing in this image?"'+
            '<br/><br/>We get a measure of the likelihood that the image contains each class in a set of classes. '+
            'The models we use to do this need to be trained using lots of examples of images '+
            'and their associated labels. We require a powerful machine learning model &#8212; called a '+
            'deep convolutional neural network &#8212; configured to extract features that predict the desired classes. '+
            'Likelihoods are based on normalized multinomial logits from a softmax classifier.'+
            '<br/><br/> This is useful for things like:'+
            '<br/><br/> * classification for monitoring and cataloging - finding and enumerating specific things in the landscape'+
            '<br/><br/> * presence/absence detection - for example, the model depicted by the image to the right '+
            'is set up to detect the presence or otherwise of a coastal barrier breach.',
            image: `${baseUrl}img/imrecog_web.png`,
            imageAlign: 'left',
            title: 'Week 1: Image recognition',
          },
        ]}
      </Block>
    );

    const Features = () => (
      <Block background="light" layout="fourColumn">
        {[
          {
            content: '"ML Mondays" is a course designed to be taught live online '+
            'to USGS scientists and researchers during Mondays in October 2020. '+
            'It is designed to teach cutting-edge deep learning techniques to scientists '+
            'whose work involves image analysis, in three main areas ...',
            image: `${baseUrl}img/undraw_react.svg`,
            imageAlign: 'top',
            title: 'Machine Learning training for USGS researchers',
          },
          {
            content: 'Classify images at the pixel level ("image segmentation"), '+
            'whole image level ("image recognition"), and object-in-image detection/classification ("object detection")',
            image: `${baseUrl}img/undraw_operating_system.svg`,
            imageAlign: 'top',
            title: '1. Image Segmentation, 2. Classification, and 3. Object Recognition',
          },
        ]}
      </Block>
    );

    const WhoIsItFor = () => (
      <Block background="dark" layout="twoColumn">
        {[
          {
            content:
            'ML-Mondays consists of 4 in-person (i.e. live, online/virtual) classes, on Oct 5, Oct 13 (a day delayed, due to the Federal Holiday Columbus Day), Oct 19, and Oct 26. '+
            'Each class starts at 10 am Pacific time (12pm Central time, 1pm Eastern time, 7am Hawaii) and lasts for up to 3 hours. '+
            '<br><br>Each class follows on from the last. Classes 1 and 4 are pairs, as are classes 2 and 3. Participants are therefore expected to last the course. '+
            'Optional homework assignments will be set for participants to carry out in their own time. '+
            '<br><br>If you cannot guarantee blocking out 3 hrs on those days in Oct, then you do not have time for the course.'+
            '<br><br>However, all course materials, including code, data, notebooks, this website, and videos, '+
            'will be made available to the entire USGS in November, after the event. Full agenda to be announced in September. '+
            '<br><br> This course is designed for USGS employees and contractors across all mission areas actively engaged '+
            'in one or more of the following topics:'+
            '<br><br> * satellite and aerial remote sensing'+
            '<br><br> * image analysis'+
            '<br><br> * geospatial analysis'+
            '<br><br> * machine learning and software development'+
            '<br><br><br><br>'+
            '**and** some experience with:'+
            '<br><br> * the python programming language (or extensive experience in any programming language, such as R, matlab, C++)'+
            '<br><br> * a command line interface such as a bash shell, '+
            'windows powershell, git bash terminal, AWS-CLI, or other terminals. ',

            image: `${baseUrl}img/undraw_questions_75e0.png`,
            imageAlign: 'right',
            title: 'Who is this course for? And when is it?',
          },
        ]}
      </Block>
    );


    const SignUp = () => (
      <Block background="light" layout="twoColumn">
        {[
          {
            content:
            'Please register your interest in attending this course, using the button below. '+
            '<br/><br/>In the event of over-subscription, the course leaders and CDI leadership '+
            'will use the information you provide to determine your potential suitability for this course.',
            image: `${baseUrl}img/undraw_programming_2svr.png`,
            imageAlign: 'top',
            title: 'How to sign up',
          },
        ]}

      </Block>
    );

    const Warning = () => (
      <Block background="dark" layout="twoColumn">
        {[
          {
            content:
            'This website and associated materials is still in active development and `alpha` version. '+
            'Use at your own risk! '+
            '<br><br> Please check back later, `watch` the github repository to receive alerts,',
            image: `${baseUrl}img/undraw_under_construction_46pa.svg`,
            imageAlign: 'top',
            title: 'Warning!',
          },
        ]}
      </Block>
    );


    const Team = () => (
      <Block background="light" layout="fourColumn">
        {[
          {
            content: 'Dan has 16 years of professional experience with scientific programming, '+
            'including 8 years using machine learning methods with imagery and geophysical data, for a variety of '+
            'measurement purposes in coastal and river science. He has worked extensively with USGS researchers, '+
            'using imagery to make measurements of sediment transport, benthic habitats, and geomorphic change. '+
            '<br/><br/>Dan is currently a contractor for the USGS Pacific Coastal and Marine Science Center, '+
            'operating through his company, Marda Science.',
            image: `${baseUrl}img/dan.jpg`,
            imageAlign: 'top',
            title: 'Course Leader: Dr Dan Buscombe',
          },
          {
            content: 'Dr Phil Wernette has experience with machine learning methods and remote sensing '+
            'and will be assisting with course planning and implementation. Phil is a Mendenhall Postdoctoral Fellow '+
            'at the Pacific Coastal and Marine Science Center in Santa Cruz, CA. '+
            '<br/><br/> Dr Leslie Hsu is the CDI coordinator and will serve as course facilitator and main contact person. '+
            '<br/><br/> Dr Jonathan Warrick is a Research Geologist at the Pacific Coastal and Marine Science Center '+
            'and will also be assisting with course planning and implementation.'+
            '<br/><br/> Other special guests will be announced soon!',
            image: `${baseUrl}img/undraw_react.svg`,
            imageAlign: 'top',
            title: 'Other team members',
          },
        ]}
      </Block>
    );

    const Info = () => (
      <Block background="dark" layout="twoColumn">
        {[
          {
            content:
            'ML Mondays is an intensive USGS course in image analysis using deep learning. '+
            'It is supported by the USGS Community for Data Integration, '+
            'in collaboration with the USGS Coastal Hazards Program.'+
            '<br><br> Deep learning is a set of methods in machine learning that use '+
            'very large neural networks to automatically extract features from imagery '+
            'then classify them. This course will assume you already know a little about '+
            'python, that you have heard of deep learning and machine learning, and you have '+
            'identified these tools as ones you would like  '+
            'to gain practical experience using together.  '+
            '<br><br> Each Monday in October 2020, Dr Daniel Buscombe, and some guests (TBD), will introduce an applied image analysis topic, '+
            'and demonstrate a technique using data and python code that have been specially curated for the course. Participants will be expected '+
            'to participate fully by carrying out the same analysis, either on the provided data or their own data. '+
            '<br><br> The course will be conducted entirely online, using Microsoft Teams and USGS Cloud Hosting Solutions, '+
            'a cloud computing environment built upon Amazon Web Services (AWS).',
            image: `${baseUrl}img/undraw_researching_22gp.png`,
            imageAlign: 'right',
            title: '',
          },
        ]}
      </Block>
    );

    const Disclaimer = () => (
      <Block background="dark" layout="twoColumn">
        {[
          {
            content:
            'This software has been approved for release by the U.S. Geological Survey (USGS). '+
            'Although the software has been subjected to rigorous review, the USGS reserves the right to update the software '+
            'as needed pursuant to further analysis and review. '+
            'No warranty, expressed or implied, is made by the USGS or the U.S. Government '+
            ' as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. Furthermore, the software is released on condition that neither the USGS '+
            'nor the U.S. Government shall be held liable for any damages resulting from its authorized or unauthorized use.',
            image: `${baseUrl}img/undraw_contract_uy56.svg`,
            imageAlign: 'top',
            title: '',
          },
        ]}
      </Block>
    );

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Info />
          <Team />
          <Warning />
          <Features />
          <WhoIsItFor />
          <ImRecog />
          <ObjRecog />
          <ImSeg />
          <ImClass />
          <SignUp />
          <HomeSplash2 siteConfig={siteConfig} language={language} />
          <Disclaimer />
        </div>
      </div>
    );
  }
}

module.exports = Index;
