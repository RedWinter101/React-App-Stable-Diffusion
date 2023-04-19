import {
  ChakraProvider,
  Heading,
  Container,
  Text,
  Input,
  Button,
  Wrap,
  Stack, 
  Image,
  SkeletonCircle,
  SkeletonText,

} from "@chakra-ui/react";
import axios from "axios";
import { useState } from "react";

const App = () => {
  const [image, updateImage] = useState();
  const [prompt, updatePrompt] = useState("");
  const [Nprompt, updateNPrompt] = useState("");
  const [inference, updateSamplingSteps] = useState("");
  const [width, updateWidth ]= useState("");
  const [height, updateHeight] = useState("");
  const [loading, updateLoading] = useState();

  const generate = async () => {
    updateLoading(true);
    //const result = await axios.get(`http://127.0.0.1:3000/?prompt=${prompt}+Nprompt=${Nprompt}+inference=${inference}+width=${width}+height=${height}`);
    const result = await axios.get(`https://a04b-35-247-4-36.ngrok-free.app/?prompt=${prompt}+Nprompt=${Nprompt}+inference=${inference}+width=${width}+height=${height}`);
    console.log(result);
    updateImage(result.data);
    updateLoading(false);
  };

  return (
    <ChakraProvider>
      <Container boxShadow={"lg"} >
        <Heading textAlign={'center'} marginTop={'10px'}>Stable DIffusion🖼️</Heading>
        <Text marginBottom={"10px"} textAlign={'justify'}>
          This react application leverages the model trained by SG161222
          to generate images using the Stable Diffusion Deep Learning
          model. 
        </Text>
        <Wrap marginBottom={"10px"} >
          <Input
            left={'10px'}
            value={prompt}
            variant={'filled'}
            placeholder="Prompt"
            onChange={(e) => updatePrompt(e.target.value)}
            width={"350px"}
          ></Input>
          <Button onClick={(e) => generate()} colorScheme={"blue"} 
          isDisabled={!prompt + !Nprompt + !inference + !width + !height}
          left={'20px'}
          >
            Generate
          </Button>
          <Input
            left={'10px'}
            value={Nprompt}
            variant={'filled'}
            placeholder="Negative Prompt"
            onChange={(e) => updateNPrompt(e.target.value)}
            width={"350px"}
          ></Input>
        </Wrap>

        <Wrap className="btn" marginBottom={"10px"}>
          <Input 
            left={'10px'}
            margin={"5px"}
            value={inference}
            placeholder="Steps"
            onChange={(e) => updateSamplingSteps(e.target.value)}
            width={"110px"} 
          ></Input>
          <Input
            left={'10px'}
            margin={"5px"}
            value={width}
            placeholder="Width"
            onChange={(e) => updateWidth(e.target.value)}
            width={"110px"} 
          ></Input>
          <Input
            left={'10px'}
            margin={"5px"}
            value={height}
            placeholder="Height"
            onChange={(e) => updateHeight(e.target.value)}
            width={"110px"} 
          ></Input>
        </Wrap>
        <Wrap boxShadow={'lg'} height={'450px'} margin={'5px'} >
        {loading ? (
          <Stack>
            <SkeletonCircle />
            <SkeletonText />
          </Stack>
        ) : image ? (
          //<Image src={`data:image/png;base64,${image}`} boxShadow="lg" />
          //<Image src={require("./lgs/download.png")} boxShadow="lg" />
          <Image src={`data:image/png;base64,${image}`} boxShadow="lg" />
        ) : null}
        </Wrap>
      </Container>
    </ChakraProvider>
  );
};

export default App;
